from __future__ import annotations

from typing import Any, Dict, Literal, Optional

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError
from pydantic import BaseModel

from server.env import RecallCoordinatorEnv
from server.models import Action, Observation, State
from server.tasks import TASKS


class ResetRequest(BaseModel):
    task_id: Literal["single_triage", "pattern_recall", "full_recall_plan"] = "single_triage"


class StepResponse(BaseModel):
    observation: Observation
    reward: float
    done: bool
    info: Dict[str, Any]


app = FastAPI(title="RecallCoordinatorEnv", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError) -> JSONResponse:
    # Hackathon requirement: return 400 {"detail":"..."} for bad input (instead of FastAPI's default 422).
    return JSONResponse(status_code=400, content={"detail": str(exc)})


def _get_env() -> RecallCoordinatorEnv:
    env: Optional[RecallCoordinatorEnv] = getattr(app.state, "env", None)
    if env is None:
        raise HTTPException(status_code=400, detail="Environment not initialized. Call POST /reset first.")
    return env


@app.post("/reset", response_model=Observation)
def reset(req: ResetRequest) -> Observation:
    if req.task_id not in TASKS:
        raise HTTPException(status_code=400, detail=f"Unknown task_id: {req.task_id}")
    env = RecallCoordinatorEnv(task_id=req.task_id, max_steps=20)
    app.state.env = env
    return env.reset()


@app.post("/step", response_model=StepResponse)
def step(action: Action) -> StepResponse:
    env = _get_env()
    try:
        obs, reward, done, info = env.step(action)
        return StepResponse(observation=obs, reward=float(reward), done=bool(done), info=info)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    except RuntimeError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    except Exception as e:
        # Keep deterministic + transparent behavior for clients.
        raise HTTPException(status_code=400, detail=f"Step failed: {e}") from e


@app.get("/state", response_model=State)
def get_state() -> State:
    env = _get_env()
    return env.state()


@app.get("/health")
def health() -> Dict[str, Any]:
    return {"status": "ok"}

