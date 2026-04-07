from __future__ import annotations

import copy
from typing import Any, Dict, List, Optional, Tuple

from server.models import Action, IncidentReport, Observation, State
from server.rewards import ALLOWED_ACTION_TYPES, RewardCalculator
from server.tasks import TASKS


# Hardcoded synthetic "database" for query_db actions. No I/O, deterministic.
# This models internal traceability data: which SKUs/batches likely affected.
SYNTHETIC_DB: Dict[str, Any] = {
    "batch": {
        # SPACE-HEATER-X: SHX-25-02 is primary suspect; SHX-25-03 partially affected
        "SPACE-HEATER-X": {"affected_batches": ["SHX-25-02"], "watch_batches": ["SHX-25-03"]},
        # AIR-FRYER-Z: AFZ-24-12 affected; AFZ-24-11 watch
        "AIR-FRYER-Z": {"affected_batches": ["AFZ-24-12"], "watch_batches": ["AFZ-24-11"]},
        # KITCH-MIX-200: single batch implicated
        "KITCH-MIX-200": {"affected_batches": ["KM200-24-11"], "watch_batches": []},
        # Toy case: bulletin, not recall, still traceable
        "TOY-CHOK-001": {"affected_batches": ["BATCH-A1"], "watch_batches": []},
    },
    "policy": {
        "severity_guidance": {
            "choking": "high",
            "electrical": "critical",
            "fire": "critical",
            "burn": "critical",
            "laceration": "high",
        }
    },
}


# Message templates and required variables, used for validation and reward shaping.
TEMPLATES: Dict[str, Dict[str, Any]] = {
    "customer_notice_v1": {
        "required": ["sku_list", "batch_list", "hazard_summary", "contact_info", "remediation_steps"],
        "channel": "customer",
    },
    "regulator_notice_v1": {
        "required": ["sku_list", "batch_list", "hazard_summary", "contact_info", "incident_count", "injury_count"],
        "channel": "regulator",
    },
    "internal_brief_v1": {
        "required": ["sku_list", "batch_list", "hazard_summary", "owners", "next_steps"],
        "channel": "internal",
    },
}


ACTION_COSTS: Dict[str, float] = {
    "query_db": 500.0,
    "draft_message": 250.0,
    "route": 100.0,
    "choose_remediation": 50.0,
    "publish_plan": 200.0,
    "classify_incident": 25.0,
}


def _deepcopy_state(s: State) -> State:
    return State.model_validate(s.model_dump(mode="python"))


def _make_observation(state: State) -> Observation:
    return Observation(
        incident_reports=state.incident_reports,
        current_plan_state=state.current_plan_state,
        constraints=state.constraints,
        validation_errors=state.validation_errors,
        step_number=state.step_number,
        task_id=state.task_id,
        task_description=state.task_description,
    )


class RecallCoordinatorEnv:
    def __init__(self, task_id: str = "single_triage", max_steps: int = 20):
        if task_id not in TASKS:
            raise ValueError(f"Unknown task_id: {task_id}")
        self.task_id = task_id
        self.max_steps = int(max_steps)
        self._reward_calc = RewardCalculator()
        self._state: Optional[State] = None

    def reset(self) -> Observation:
        spec = TASKS[self.task_id]

        # Deterministic initial state.
        self._state = State(
            incident_reports=copy.deepcopy(spec.initial_reports),
            current_plan_state={
                "tickets": [],
                "drafts": [],
                "approvals": [],
                "queries": [],
                "action_history": [],
            },
            constraints=dict(spec.initial_constraints),
            validation_errors=[],
            step_number=0,
            task_id=spec.task_id,
            task_description=spec.description,
            classified_reports={},
            routed_teams=[],
            drafted_messages=[],
            chosen_remediation=None,
            plan_published=False,
            errors_made=[],
            total_reward_so_far=0.0,
        )
        return _make_observation(self._state)

    def state(self) -> State:
        if self._state is None:
            raise RuntimeError("Environment not reset. Call reset() first.")
        return _deepcopy_state(self._state)

    def _record_action_history(self, action: Action) -> None:
        assert self._state is not None
        hist = self._state.current_plan_state.get("action_history")
        if not isinstance(hist, list):
            self._state.current_plan_state["action_history"] = []
            hist = self._state.current_plan_state["action_history"]
        hist.append({"signature": (action.action_type, action.parameters)})

    def _spend_budget(self, action_type: str) -> None:
        assert self._state is not None
        cost = float(ACTION_COSTS.get(action_type, 0.0))
        budget = self._state.constraints.get("budget_remaining", 0.0)
        if isinstance(budget, (int, float)):
            self._state.constraints["budget_remaining"] = float(budget) - cost

    def _advance_deadline(self) -> None:
        # Each step consumes 1 hour of "work time" deterministically.
        assert self._state is not None
        deadline_hours = self._state.constraints.get("deadline_hours")
        if isinstance(deadline_hours, int):
            self._state.constraints["deadline_hours"] = max(0, deadline_hours - 1)

    def _validate_action(self, action: Action) -> List[str]:
        errors: List[str] = []
        if action.action_type not in ALLOWED_ACTION_TYPES:
            errors.append(f"Invalid action_type: {action.action_type}")
            return errors

        p = action.parameters or {}
        if not isinstance(p, dict):
            errors.append("parameters must be an object/dict")
            return errors

        required: Dict[str, List[str]] = {
            "classify_incident": ["report_id", "severity", "hazard_type"],
            "route": ["team"],
            "query_db": ["entity", "filters"],
            "draft_message": ["channel", "template_id", "variables"],
            "choose_remediation": ["strategy"],
            "publish_plan": ["plan_id"],
        }
        for k in required[action.action_type]:
            if k not in p:
                errors.append(f"Missing parameter: {k}")

        if action.action_type == "classify_incident":
            sev = str(p.get("severity", "")).lower().strip()
            if sev not in {"low", "medium", "high", "critical"}:
                errors.append("Invalid severity (must be low|medium|high|critical)")
            if not str(p.get("hazard_type", "")).strip():
                errors.append("hazard_type must be non-empty")
        if action.action_type == "route":
            team = str(p.get("team", "")).lower().strip()
            if team not in {"legal", "quality", "comms", "ops"}:
                errors.append("Invalid team (must be legal|quality|comms|ops)")
        if action.action_type == "query_db":
            if not isinstance(p.get("filters", None), dict):
                errors.append("filters must be a dict")
            if not str(p.get("entity", "")).strip():
                errors.append("entity must be non-empty")
        if action.action_type == "draft_message":
            channel = str(p.get("channel", "")).lower().strip()
            if channel not in {"customer", "regulator", "internal"}:
                errors.append("Invalid channel (must be customer|regulator|internal)")
            template_id = str(p.get("template_id", "")).strip()
            if template_id not in TEMPLATES:
                errors.append(f"Unknown template_id: {template_id}")
            if not isinstance(p.get("variables", None), dict):
                errors.append("variables must be a dict")
        if action.action_type == "choose_remediation":
            strategy = str(p.get("strategy", "")).lower().strip()
            if strategy not in {"repair", "replace", "refund", "service_bulletin", "recall"}:
                errors.append("Invalid strategy")
        if action.action_type == "publish_plan":
            if not str(p.get("plan_id", "")).strip():
                errors.append("plan_id must be non-empty")
        return errors

    def _apply_action(self, action: Action) -> Dict[str, Any]:
        assert self._state is not None
        p = action.parameters or {}
        result: Dict[str, Any] = {"ok": False}

        if action.action_type == "classify_incident":
            rid = str(p.get("report_id", "")).strip()
            severity = str(p.get("severity", "")).lower().strip()
            hazard_type = str(p.get("hazard_type", "")).strip()
            known_ids = {r.report_id for r in self._state.incident_reports}
            if rid in known_ids:
                self._state.classified_reports[rid] = {"severity": severity, "hazard_type": hazard_type}
                result.update({"ok": True})
            else:
                result.update({"ok": False, "reason": "unknown_report_id"})

        elif action.action_type == "route":
            team = str(p.get("team", "")).lower().strip()
            if team in {"legal", "quality", "comms", "ops"}:
                new_team = team not in self._state.routed_teams
                if new_team:
                    self._state.routed_teams.append(team)
                self._state.current_plan_state.setdefault("tickets", []).append({"team": team, "status": "queued"})
                result.update({"ok": True, "new_team": new_team})

        elif action.action_type == "query_db":
            entity = str(p.get("entity", "")).lower().strip()
            filters = p.get("filters", {})
            response: Dict[str, Any] = {"entity": entity, "filters": filters, "results": []}

            # Supported deterministic query patterns:
            # - entity contains "batch": filters may contain {"sku": "..."} or {"skus": ["..."]}
            if "batch" in entity:
                skus: List[str] = []
                if isinstance(filters, dict):
                    if isinstance(filters.get("sku"), str):
                        skus = [filters["sku"]]
                    elif isinstance(filters.get("skus"), list):
                        skus = [str(x) for x in filters.get("skus", [])]
                for sku in skus:
                    trace = SYNTHETIC_DB["batch"].get(sku)
                    if trace:
                        response["results"].append(
                            {"product_sku": sku, "affected_batches": trace["affected_batches"], "watch_batches": trace["watch_batches"]}
                        )

            self._state.current_plan_state.setdefault("queries", []).append({"entity": entity, "filters": filters, "response": response})
            result.update({"ok": True, "response": response})

        elif action.action_type == "draft_message":
            channel = str(p.get("channel", "")).lower().strip()
            template_id = str(p.get("template_id", "")).strip()
            variables = p.get("variables", {})

            template = TEMPLATES.get(template_id)
            if template and template["channel"] == channel and isinstance(variables, dict):
                required = template["required"]
                missing = [k for k in required if not str(variables.get(k, "")).strip()]
                template_valid = len(missing) == 0

                msg = {
                    "channel": channel,
                    "template_id": template_id,
                    "variables": variables,
                    "missing": missing,
                }
                self._state.drafted_messages.append(msg)
                self._state.current_plan_state.setdefault("drafts", []).append({"channel": channel, "template_id": template_id})
                result.update({"ok": True, "template_valid": template_valid, "missing": missing})
            else:
                result.update({"ok": False, "reason": "template_channel_mismatch"})

        elif action.action_type == "choose_remediation":
            strategy = str(p.get("strategy", "")).lower().strip()
            prerequisite_met = len(self._state.classified_reports) > 0
            if strategy in {"repair", "replace", "refund", "service_bulletin", "recall"}:
                self._state.chosen_remediation = strategy
                result.update({"ok": True, "prerequisite_met": prerequisite_met})
            else:
                result.update({"ok": False, "reason": "invalid_strategy"})

        elif action.action_type == "publish_plan":
            plan_id = str(p.get("plan_id", "")).strip()
            if len(self._state.classified_reports) == 0:
                result.update({"ok": False, "reason": "no_classifications"})
            else:
                requirements_met = self._check_publish_requirements()
                self._state.current_plan_state.setdefault("approvals", []).append({"plan_id": plan_id, "status": "submitted"})
                if requirements_met:
                    self._state.plan_published = True
                    result.update({"ok": True, "requirements_met": True})
                else:
                    result.update({"ok": True, "requirements_met": False})

        return result

    def _check_publish_requirements(self) -> bool:
        assert self._state is not None
        # Minimal gating to enable publish reward shaping:
        # - at least one classification
        # - remediation chosen
        if len(self._state.classified_reports) == 0:
            return False
        if not self._state.chosen_remediation:
            return False
        # For hard task, require all 4 teams routed and at least 3 drafted messages.
        if self._state.task_id == "full_recall_plan":
            if not all(t in self._state.routed_teams for t in ["legal", "quality", "comms", "ops"]):
                return False
            channels = {m.get("channel") for m in self._state.drafted_messages if isinstance(m, dict)}
            if not {"customer", "regulator", "internal"}.issubset(channels):
                return False
            # Require at least one query_db call for batches identification.
            queries = self._state.current_plan_state.get("queries", [])
            if not (isinstance(queries, list) and len(queries) > 0):
                return False
        return True

    def step(self, action: Action) -> Tuple[Observation, float, bool, Dict[str, Any]]:
        if self._state is None:
            raise RuntimeError("Environment not reset. Call reset() first.")

        self._state.validation_errors = []
        validation_errors = self._validate_action(action)
        if validation_errors:
            self._state.validation_errors = validation_errors
            self._state.errors_made.extend(validation_errors)
            self._record_action_history(action)
            # still advance time/step to avoid infinite loops
            self._state.step_number += 1
            self._advance_deadline()
            step_reward = self._reward_calc.calculate_step_reward(action, {"ok": False}, self._state)
            self._state.total_reward_so_far += step_reward
            done = self._state.step_number >= self.max_steps or self._state.plan_published is True
            info: Dict[str, Any] = {"grader_score": 0.0, "error": "; ".join(validation_errors)}
            if done:
                hist = self._state.current_plan_state.get("action_history", [])
                episode_info = {
                    "task_id": self._state.task_id,
                    "steps_taken": self._state.step_number,
                    "max_steps": self.max_steps,
                    "budget_remaining": self._state.constraints.get("budget_remaining"),
                    "deadline_hours": self._state.constraints.get("deadline_hours"),
                }
                grader_score = TASKS[self._state.task_id].grader(self._state, hist if isinstance(hist, list) else [], episode_info)
                final_reward = self._reward_calc.calculate_final_reward(
                    task_completed=self._state.plan_published,
                    steps_taken=self._state.step_number,
                    errors=self._state.errors_made,
                )
                info["grader_score"] = grader_score
                info["final_reward"] = final_reward
            return _make_observation(self._state), step_reward, done, info

        # Apply action side effects.
        self._record_action_history(action)
        self._spend_budget(action.action_type)
        result = self._apply_action(action)

        # Update step counters and time.
        self._state.step_number += 1
        self._advance_deadline()

        # Calculate reward.
        step_reward = self._reward_calc.calculate_step_reward(action, result, self._state)
        self._state.total_reward_so_far += step_reward

        done = self._state.step_number >= self.max_steps or self._state.plan_published is True
        info: Dict[str, Any] = {"grader_score": 0.0, "error": None}

        if done:
            hist = self._state.current_plan_state.get("action_history", [])
            episode_info = {
                "task_id": self._state.task_id,
                "steps_taken": self._state.step_number,
                "max_steps": self.max_steps,
                "budget_remaining": self._state.constraints.get("budget_remaining"),
                "deadline_hours": self._state.constraints.get("deadline_hours"),
            }
            grader_score = TASKS[self._state.task_id].grader(self._state, hist if isinstance(hist, list) else [], episode_info)
            final_reward = self._reward_calc.calculate_final_reward(
                task_completed=self._state.plan_published,
                steps_taken=self._state.step_number,
                errors=self._state.errors_made,
            )
            info["grader_score"] = grader_score
            info["final_reward"] = final_reward
        return _make_observation(self._state), step_reward, done, info

