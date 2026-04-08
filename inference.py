from __future__ import annotations

import json
import os
import time
from typing import Any, Dict, List, Optional

import httpx
from openai import OpenAI


API_BASE_URL = os.getenv("API_BASE_URL", "https://api.openai.com/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "gpt-4.1-mini")
HF_TOKEN = os.getenv("HF_TOKEN")
if HF_TOKEN is None:
    raise ValueError("HF_TOKEN environment variable is required")

ENV_URL = os.getenv("ENV_URL", "http://localhost:7860")


SYSTEM_PROMPT = (
    "You are a product safety recall coordinator agent. \n"
    "You will receive the current environment observation as JSON. \n"
    "You must respond with ONLY a valid JSON object representing your next action.\n"
    "Valid action_types: classify_incident, route, query_db, draft_message, \n"
    "choose_remediation, publish_plan.\n"
    "Each action must have: {\"action_type\": \"...\", \"parameters\": {...}}\n"
    "Do not include any explanation or text outside the JSON object."
)


def user_prompt(observation_json: str, task_description: str) -> str:
    return (
        "Current observation:\n"
        f"{observation_json}\n\n"
        f"Choose the single best next action to complete task: {task_description}\n"
        "Respond with JSON only."
    )


def _json_dumps_compact(obj: Any) -> str:
    return json.dumps(obj, separators=(",", ":"), ensure_ascii=False)


def _parse_action_or_default(text: str) -> Dict[str, Any]:
    try:
        data = json.loads(text)
        if isinstance(data, dict) and "action_type" in data and "parameters" in data and isinstance(data["parameters"], dict):
            return data
    except Exception:
        pass
    return {
        "action_type": "classify_incident",
        "parameters": {"report_id": "r1", "severity": "high", "hazard_type": "unknown"},
    }


def _fmt_done(done: bool) -> str:
    return "true" if done else "false"


def _fmt_error(err: Optional[str]) -> str:
    if err is None:
        return "null"
    return err.replace("\n", " ").strip()


def main() -> None:
    client = OpenAI(base_url=API_BASE_URL, api_key=HF_TOKEN)

    tasks = ["single_triage", "pattern_recall", "full_recall_plan"]
    timeout = httpx.Timeout(20.0, connect=10.0)

    with httpx.Client(timeout=timeout) as http:
        for task_id in tasks:
            info: Dict[str, Any] = {}
            final_grader_score = 0.0

            # Reset env
            obs: Dict[str, Any] = {}
            try:
                r = http.post(f"{ENV_URL}/reset", json={"task_id": task_id})
                r.raise_for_status()
                obs = r.json()
            except Exception as e:
                # If reset fails, still print required header and attempt steps (they will fail).
                obs = {"task_id": task_id, "task_description": "", "step_number": 0, "validation_errors": [str(e)]}

            print(f"[START] task={task_id} env=RecallCoordinatorEnv model={MODEL_NAME}", flush=True)

            rewards: List[float] = []
            success = False
            steps_taken = 0

            for step_idx in range(1, 21):
                steps_taken = step_idx
                observation_json = _json_dumps_compact(obs)
                task_description = str(obs.get("task_description", ""))
                prompt = user_prompt(observation_json, task_description)

                # Ask model for next action JSON
                action_obj: Dict[str, Any]
                try:
                    resp = client.chat.completions.create(
                        model=MODEL_NAME,
                        messages=[
                            {"role": "system", "content": SYSTEM_PROMPT},
                            {"role": "user", "content": prompt},
                        ],
                        temperature=0.2,
                        max_tokens=300,
                    )
                    content = resp.choices[0].message.content or ""
                    action_obj = _parse_action_or_default(content)
                except Exception:
                    action_obj = {
                        "action_type": "classify_incident",
                        "parameters": {"report_id": "r1", "severity": "high", "hazard_type": "unknown"},
                    }

                action_type = str(action_obj.get("action_type", ""))

                # Step env
                reward = 0.0
                done = False
                err: Optional[str] = None
                info: Dict[str, Any] = {}
                try:
                    sr = http.post(f"{ENV_URL}/step", json=action_obj)
                    sr.raise_for_status()
                    payload = sr.json()
                    obs = payload.get("observation", {})
                    reward = float(payload.get("reward", 0.0))
                    done = bool(payload.get("done", False))
                    info = payload.get("info", {}) or {}
                except Exception as e:
                    err = str(e)
                    reward = 0.0
                    done = False

                rewards.append(reward)
                print(
                    f"[STEP] step={step_idx} action={action_type} reward={reward:.2f} done={_fmt_done(done)} error={_fmt_error(info.get('error') if err is None else err)}",
                    flush=True,
                )

                if done:
                    final_grader_score = float((info or {}).get("grader_score", 0.0))
                    success = final_grader_score >= 0.9
                    break

                # Keep runtime bounded and allow server time.
                time.sleep(0.05)

            rewards_str = ",".join(f"{r:.2f}" for r in rewards)
            final_grader_score = float((info or {}).get("grader_score", 0.0))
            print(
                f"[END] success={_fmt_done(success)} steps={steps_taken} score={final_grader_score:.3f} rewards={rewards_str}",
                flush=True,
            )


if __name__ == "__main__":
    main()

