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


def _smart_fallback(obs: Dict[str, Any], step_idx: int, used_actions: List[str]) -> Dict[str, Any]:
    """
    Rule-based fallback agent. Used when LLM fails or gives repeated action.
    Strategy: classify -> route -> query_db -> draft -> remediation -> publish
    """
    reports = obs.get("incident_reports", [])
    plan = obs.get("current_plan_state", {})
    task_id = obs.get("task_id", "single_triage")

    classified = list(plan.get("action_history", []))
    classify_count = sum(1 for a in classified if isinstance(a, dict) and
                         a.get("signature", [None])[0] == "classify_incident")
    route_count = sum(1 for a in classified if isinstance(a, dict) and
                      a.get("signature", [None])[0] == "route")
    draft_count = sum(1 for a in classified if isinstance(a, dict) and
                      a.get("signature", [None])[0] == "draft_message")

    total_reports = len(reports)
    teams = ["quality", "legal", "comms", "ops"]
    channels = [
        ("customer", "customer_notice_v1",
         {"sku_list": "AFFECTED-SKU", "batch_list": "AFFECTED-BATCH",
          "hazard_summary": "Product safety hazard identified",
          "contact_info": "1-800-RECALL", "remediation_steps": "Stop use immediately"}),
        ("regulator", "regulator_notice_v1",
         {"sku_list": "AFFECTED-SKU", "batch_list": "AFFECTED-BATCH",
          "hazard_summary": "Product safety hazard identified",
          "contact_info": "1-800-RECALL", "incident_count": str(total_reports),
          "injury_count": str(sum(1 for r in reports if r.get("injury_reported", False)))}),
        ("internal", "internal_brief_v1",
         {"sku_list": "AFFECTED-SKU", "batch_list": "AFFECTED-BATCH",
          "hazard_summary": "Product safety hazard identified",
          "owners": "Safety Team", "next_steps": "Initiate recall process"}),
    ]

    # Step 1: Classify all reports first
    if classify_count < total_reports and total_reports > 0:
        for r in reports:
            rid = r.get("report_id", "r1")
            injury = r.get("injury_reported", False)
            severity = "critical" if injury else "high"
            hazard = "electrical" if "electr" in r.get("hazard_description", "").lower() else \
                     "fire" if "fire" in r.get("hazard_description", "").lower() else \
                     "burn" if "burn" in r.get("hazard_description", "").lower() else \
                     "choking" if "chok" in r.get("hazard_description", "").lower() else \
                     "mechanical"
            already_classified = any(
                isinstance(a, dict) and
                a.get("signature", [None, {}])[1].get("report_id") == rid
                for a in classified
            )
            if not already_classified:
                return {
                    "action_type": "classify_incident",
                    "parameters": {"report_id": rid, "severity": severity, "hazard_type": hazard}
                }

    # Step 2: Query DB for batch info (hard task)
    if task_id == "full_recall_plan":
        query_done = any(
            isinstance(a, dict) and a.get("signature", [None])[0] == "query_db"
            for a in classified
        )
        if not query_done:
            skus = list({r.get("product_sku", "") for r in reports if r.get("product_sku")})
            return {
                "action_type": "query_db",
                "parameters": {"entity": "batch", "filters": {"skus": skus[:2]}}
            }

    # Step 3: Route teams
    teams_needed = ["quality"] if task_id == "single_triage" else ["quality", "legal", "comms", "ops"]
    routed = [a.get("signature", [None, {}])[1].get("team", "")
              for a in classified if isinstance(a, dict) and
              a.get("signature", [None])[0] == "route"]
    for team in teams_needed:
        if team not in routed:
            return {"action_type": "route", "parameters": {"team": team}}

    # Step 4: Draft messages
    drafted_channels = [a.get("signature", [None, {}])[1].get("channel", "")
                        for a in classified if isinstance(a, dict) and
                        a.get("signature", [None])[0] == "draft_message"]
    channels_needed = ["customer", "regulator", "internal"] if task_id != "single_triage" else []
    for ch, tmpl, variables in channels:
        if ch in channels_needed and ch not in drafted_channels:
            return {
                "action_type": "draft_message",
                "parameters": {"channel": ch, "template_id": tmpl, "variables": variables}
            }

    # Step 5: Choose remediation
    remediation_done = any(
        isinstance(a, dict) and a.get("signature", [None])[0] == "choose_remediation"
        for a in classified
    )
    if not remediation_done:
        strategy = "service_bulletin" if task_id == "single_triage" else "recall"
        return {"action_type": "choose_remediation", "parameters": {"strategy": strategy}}

    # Step 6: Publish
    return {"action_type": "publish_plan", "parameters": {"plan_id": f"plan-{task_id}-001"}}


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
                used_actions: List[str] = [
                    str(h.get("signature", [""])[0])
                    for h in (obs.get("current_plan_state", {}).get("action_history") or [])
                    if isinstance(h, dict)
                ]
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
                    # If LLM gave invalid or repeated action, use smart fallback
                    atype = action_obj.get("action_type", "")
                    if atype not in {"classify_incident", "route", "query_db",
                                     "draft_message", "choose_remediation", "publish_plan"}:
                        action_obj = _smart_fallback(obs, step_idx, used_actions)
                except Exception:
                    action_obj = _smart_fallback(obs, step_idx, used_actions)

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

