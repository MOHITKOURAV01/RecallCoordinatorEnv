from __future__ import annotations

from typing import Any, Dict, Literal, Optional

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse
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


@app.get("/", response_class=HTMLResponse)
def root() -> str:
    """HF Spaces opens `/` in the browser; API-only apps had no route here (404)."""
    return """<!DOCTYPE html>
<html lang="en">
<head><meta charset="utf-8"/><meta name="viewport" content="width=device-width, initial-scale=1"/>
<title>RecallCoordinatorEnv</title>
<style>
body{font-family:system-ui,sans-serif;max-width:42rem;margin:2rem auto;padding:0 1rem;line-height:1.5;color:#1a1a1a}
a{color:#0b57d0} code{background:#f2f2f2;padding:.1rem .35rem;border-radius:4px}
ul{padding-left:1.2rem}
</style>
</head>
<body>
<h1>RecallCoordinatorEnv</h1>
<p>OpenEnv API is running. This Space exposes a <strong>REST API</strong> (no web UI on <code>/</code> before).</p>
<ul>
<li><a href="/docs">Interactive API docs (Swagger)</a></li>
<li><a href="/health">GET /health</a> — liveness + task ids</li>
<li><a href="/tasks">GET /tasks</a> — task catalog</li>
<li><a href="/episode_summary">GET /episode_summary</a> — live episode progress + grader score</li>
</ul>
<p>Use <code>POST /reset</code> then <code>POST /step</code> with JSON bodies (see <code>/docs</code>).</p>
</body>
</html>"""


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
def reset(req: Optional[ResetRequest] = None) -> Observation:
    task_id = (req.task_id if req is not None else None) or "single_triage"
    if task_id not in TASKS:
        raise HTTPException(status_code=400, detail=f"Unknown task_id: {task_id}")
    env = RecallCoordinatorEnv(task_id=task_id, max_steps=20)
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
    return {
        "status": "ok",
        "tasks": ["single_triage", "pattern_recall", "full_recall_plan"],
        "version": "1.0.0",
        "env": "RecallCoordinatorEnv",
    }


@app.get("/tasks")
def list_tasks() -> Dict[str, Any]:
    return {
        "tasks": [
            {
                "task_id": spec.task_id,
                "difficulty": spec.difficulty,
                "description": spec.description,
                "num_reports": len(spec.initial_reports),
                "constraints": dict(spec.initial_constraints),
            }
            for spec in TASKS.values()
        ]
    }


@app.post("/validate")
def validate_action(action: Action) -> Dict[str, Any]:
    from server.rewards import ALLOWED_ACTION_TYPES

    errors = []
    if action.action_type not in ALLOWED_ACTION_TYPES:
        errors.append(
            f"Invalid action_type: '{action.action_type}'. "
            f"Must be one of: {sorted(ALLOWED_ACTION_TYPES)}"
        )
    p = action.parameters or {}
    required_params = {
        "classify_incident": ["report_id", "severity", "hazard_type"],
        "route": ["team"],
        "query_db": ["entity", "filters"],
        "draft_message": ["channel", "template_id", "variables"],
        "choose_remediation": ["strategy"],
        "publish_plan": ["plan_id"],
    }
    if action.action_type in required_params:
        for k in required_params[action.action_type]:
            if k not in p:
                errors.append(f"Missing required parameter: '{k}'")
    return {
        "valid": len(errors) == 0,
        "errors": errors,
        "action_type": action.action_type,
    }


@app.get("/episode_summary")
def episode_summary() -> Dict[str, Any]:
    """Return current episode progress summary with grader score estimate."""
    env = getattr(app.state, "env", None)
    if env is None:
        return {
            "initialized": False,
            "message": "Call POST /reset to start an episode",
        }
    state = env.state()
    from server.tasks import TASKS
    task_spec = TASKS.get(state.task_id)
    hist = state.current_plan_state.get("action_history", [])
    episode_info = {
        "task_id": state.task_id,
        "steps_taken": state.step_number,
        "budget_remaining": state.constraints.get("budget_remaining"),
        "deadline_hours": state.constraints.get("deadline_hours"),
    }
    grader_score = 0.0
    if task_spec:
        try:
            grader_score = task_spec.grader(state, hist, episode_info)
        except Exception:
            grader_score = 0.0
    return {
        "initialized": True,
        "task_id": state.task_id,
        "step_number": state.step_number,
        "plan_published": state.plan_published,
        "classified_count": len(state.classified_reports),
        "total_reports": len(state.incident_reports),
        "routed_teams": state.routed_teams,
        "chosen_remediation": state.chosen_remediation,
        "drafted_channels": list({
            m.get("channel") for m in state.drafted_messages 
            if isinstance(m, dict) and m.get("channel")
        }),
        "errors_count": len(state.errors_made),
        "total_reward_so_far": round(state.total_reward_so_far, 3),
        "current_grader_score": round(grader_score, 3),
        "budget_remaining": state.constraints.get("budget_remaining"),
        "deadline_hours": state.constraints.get("deadline_hours"),
    }

