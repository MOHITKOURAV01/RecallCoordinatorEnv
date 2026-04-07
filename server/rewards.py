from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

from server.models import Action, State


ALLOWED_ACTION_TYPES = {
    "classify_incident",
    "route",
    "query_db",
    "draft_message",
    "choose_remediation",
    "publish_plan",
}


def _clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


def _sig(action: Action) -> Tuple[str, Tuple[Tuple[str, Any], ...]]:
    """
    Stable action signature for loop detection.

    We normalize the parameters dict into a sorted tuple to avoid dict-order noise.
    """
    params = action.parameters if isinstance(action.parameters, dict) else {}
    return (str(action.action_type), tuple(sorted(params.items(), key=lambda kv: str(kv[0]))))


def _count_filled_vars(variables: Any) -> Tuple[int, int]:
    """Return (#filled, #total) for a template variables dict-like object."""
    if not isinstance(variables, dict):
        return (0, 0)
    total = len(variables)
    filled = 0
    for _, v in variables.items():
        if isinstance(v, (int, float)) and v != 0:
            filled += 1
        elif isinstance(v, bool):
            filled += 1
        elif v is not None and str(v).strip():
            filled += 1
    return (filled, total)


class RewardCalculator:
    """
    Judge-facing shaping design:

    - Trajectory-level signals:
      - Micro-rewards for syntactically valid, *useful* actions (classification, routing, drafting).
      - Milestone rewards when meaningful subgoals are reached (e.g., all reports classified).
      - Final completion bonus handled by calculate_final_reward().

    - Penalty design:
      - Loop/redundancy penalties using action signature frequency.
      - Invalid-action penalties (spec violations, missing params) via state.validation_errors/result.
      - Time-waste penalties after a reasonable threshold (task-dependent).
      - "Destructive" penalties for irreversible bad sequencing (e.g., publishing too early).

    - Normalization:
      - Step reward is always clamped to [-0.5, 1.0] (requirement).
      - Final episode score is always clamped to [0.0, 1.0] (requirement).
    """

    def calculate_step_reward(self, action: Action, result: Dict[str, Any], state: State) -> float:
        reward = 0.0

        # -------------------------
        # 1) Invalid / spec errors
        # -------------------------
        # Penalize invalid action_types immediately (hard spec violation).
        if action.action_type not in ALLOWED_ACTION_TYPES:
            reward -= 0.15

        # Penalize validation errors recorded by env (missing params, invalid enums, etc.).
        if isinstance(state.validation_errors, list) and state.validation_errors:
            # Strong but not catastrophic: agent can recover.
            reward -= min(0.25, 0.05 * len(state.validation_errors))

        # ------------------------------------
        # 2) Loop / redundancy (trajectory)
        # ------------------------------------
        # Detect repeated signatures and penalize escalating with frequency.
        hist = state.current_plan_state.get("action_history", [])
        if isinstance(hist, list) and hist:
            signature = _sig(action)
            sigs: List[Tuple[str, Tuple[Tuple[str, Any], ...]]] = []
            for h in hist:
                if isinstance(h, dict) and isinstance(h.get("signature"), tuple) and len(h["signature"]) == 2:
                    # Stored as (action_type, params_dict) in env; normalize safely.
                    at, params = h["signature"]
                    if isinstance(params, dict):
                        sigs.append((str(at), tuple(sorted(params.items(), key=lambda kv: str(kv[0])))))
            freq = sum(1 for s in sigs if s == signature)
            if freq >= 2:
                # Escalating penalty for loops: 2nd repeat small, then larger.
                reward -= min(0.20, 0.03 * (freq - 1))

            # Oscillation penalty: alternating between two signatures in last 4 steps.
            if len(sigs) >= 4:
                last4 = sigs[-4:]
                if last4[0] == last4[2] and last4[1] == last4[3] and last4[0] != last4[1]:
                    reward -= 0.08

        # ------------------------------------
        # 3) Time-waste beyond threshold
        # ------------------------------------
        # Task-dependent "reasonable" step counts (encourages efficient completion).
        # This is a soft penalty: it nudges planning without dominating.
        thresholds = {"single_triage": 6, "pattern_recall": 12, "full_recall_plan": 16}
        thr = thresholds.get(state.task_id, 12)
        if isinstance(state.step_number, int) and state.step_number > thr:
            reward -= 0.01 * (state.step_number - thr)

        # ------------------------------------
        # 4) Positive micro-rewards per action
        # ------------------------------------
        atype = str(action.action_type)
        ok = bool(result.get("ok") is True)

        if atype == "classify_incident":
            # Micro-reward for useful classification (valid severity + non-empty hazard_type).
            sev = str(action.parameters.get("severity", "")).strip().lower()
            hazard_type = str(action.parameters.get("hazard_type", "")).strip()
            if sev in {"low", "medium", "high", "critical"} and hazard_type and ok:
                reward += 0.08

            # Milestone shaping: partial credit proportional to coverage of classifications.
            # This creates distinct reward levels for medium/hard tasks.
            total = len(state.incident_reports) if isinstance(state.incident_reports, list) else 0
            done_n = len(state.classified_reports) if isinstance(state.classified_reports, dict) else 0
            if total > 0:
                coverage = done_n / total
                # small dense signal as coverage increases
                reward += 0.04 * coverage
                # milestone bumps at key levels (meets requirement for multiple distinct levels)
                if abs(coverage - 1.0) < 1e-9:
                    reward += 0.10
                elif coverage >= 0.75:
                    reward += 0.05
                elif coverage >= 0.25:
                    reward += 0.02

        elif atype == "route":
            # Reward routing new teams; discourage spamming the same team repeatedly.
            if ok and result.get("new_team") is True:
                reward += 0.07
            elif ok and result.get("new_team") is False:
                reward -= 0.02

            # Milestone: for hard task, routing all 4 teams is a major sub-goal.
            if state.task_id == "full_recall_plan":
                teams = set(state.routed_teams or [])
                # granular levels (>=5): 1,2,3,4 teams routed
                reward += 0.02 * len(teams)
                if teams.issuperset({"legal", "quality", "comms", "ops"}):
                    reward += 0.08

        elif atype == "query_db":
            # Reward "information seeking" that is likely to reduce uncertainty.
            # Useful if it includes sku/skus filter and yields results.
            filters = action.parameters.get("filters")
            has_sku_filter = isinstance(filters, dict) and (
                isinstance(filters.get("sku"), str) or isinstance(filters.get("skus"), list)
            )
            has_results = isinstance(result.get("response", {}).get("results"), list) and len(result["response"]["results"]) > 0
            if ok and has_sku_filter and has_results:
                reward += 0.10
            elif ok and has_sku_filter:
                reward += 0.04
            elif ok:
                reward += 0.01

        elif atype == "draft_message":
            # Reward drafting messages, but strongly prefer completeness (filled required variables).
            template_valid = bool(result.get("template_valid") is True)
            if ok and template_valid:
                reward += 0.12
            elif ok:
                # Partial credit: count filled variables to create non-binary gradient.
                filled, total = _count_filled_vars(action.parameters.get("variables"))
                if total > 0:
                    reward += 0.06 * (filled / total)
                else:
                    reward += 0.01

            # Milestones: for medium/hard, drafting the right channels matters.
            channels = {m.get("channel") for m in (state.drafted_messages or []) if isinstance(m, dict)}
            if state.task_id in {"pattern_recall", "full_recall_plan"}:
                if "customer" in channels:
                    reward += 0.03
                if "regulator" in channels:
                    reward += 0.03
            if state.task_id == "full_recall_plan" and "internal" in channels:
                reward += 0.02

        elif atype == "choose_remediation":
            # Encourage correct sequencing: choose remediation after *some* classification.
            prerequisite_met = bool(result.get("prerequisite_met") is True)
            if ok and prerequisite_met:
                reward += 0.10
            elif ok and not prerequisite_met:
                # Destructive sequencing: committing to strategy without evidence.
                reward -= 0.10

            # Task-aware target shaping (not the grader—just guidance).
            strategy = str(action.parameters.get("strategy", "")).strip().lower()
            if state.task_id == "single_triage" and strategy == "service_bulletin":
                reward += 0.08  # intermediate signal for easy task (requirement)
            if state.task_id in {"pattern_recall", "full_recall_plan"} and strategy == "recall":
                reward += 0.08

            # Penalize flip-flopping remediation decisions (irreversible operational churn).
            prev = state.chosen_remediation
            if isinstance(prev, str) and prev and prev != strategy:
                reward -= 0.12

        elif atype == "publish_plan":
            # Publishing is treated as an irreversible action in this workflow.
            # Reward only if requirements are met; penalize premature publish attempts.
            requirements_met = bool(result.get("requirements_met") is True)
            if ok and requirements_met:
                reward += 0.25  # milestone reward
            elif ok and not requirements_met:
                reward -= 0.12
            elif result.get("reason") == "no_classifications":
                reward -= 0.20

        # ------------------------------------
        # 5) Constraint pressure shaping
        # ------------------------------------
        # Budget: small penalty when nearing zero; larger penalty if exceeded.
        budget_remaining = state.constraints.get("budget_remaining")
        if isinstance(budget_remaining, (int, float)):
            b = float(budget_remaining)
            if b < 0.0:
                reward -= 0.25
            elif b < 1000.0:
                reward -= 0.03

        # Deadline: encourage finishing before it hits 0.
        deadline_hours = state.constraints.get("deadline_hours")
        if isinstance(deadline_hours, int):
            if deadline_hours == 0 and not state.plan_published:
                reward -= 0.10

        # Normalize step reward to required range.
        return _clamp(reward, -0.5, 1.0)

    def calculate_final_reward(self, task_completed: bool, steps_taken: int, errors: List[str]) -> float:
        """
        Final episode score in [0.0, 1.0], separate from step rewards.

        We keep it intentionally simple and stable for benchmarking:
        - Base is 1.0 if task is completed (plan published), else 0.0.
        - Efficiency bonus if solved quickly.
        - Clean-run bonus if no errors.
        - Penalty if too many errors or very inefficient.

        Note: In this repo, deterministic task graders are returned in the FastAPI `info`
        dict for evaluation. This function is a *completion-quality* bonus layer.
        """
        score = 1.0 if bool(task_completed) else 0.0

        # Efficiency: encourage short-horizon planning without making it brittle.
        if steps_taken <= 8:
            score += 0.05
        elif steps_taken >= 18:
            score -= 0.05

        # Clean run: avoid invalid actions/loops.
        if len(errors) == 0:
            score += 0.05
        else:
            # Mild penalty that scales but won't dominate.
            score -= min(0.20, 0.02 * len(errors))

        return _clamp(score, 0.0, 1.0)


# ---------------------------------------------------------------------------
# Unit-test-like examples (run as a script)
# ---------------------------------------------------------------------------
def _make_min_state(task_id: str = "single_triage") -> State:
    """
    Minimal State constructor for reward examples.
    We only fill the fields that rewards.py reads.
    """
    return State(
        incident_reports=[],
        current_plan_state={"action_history": []},
        constraints={"budget_remaining": 10000.0, "deadline_hours": 72, "regulator_deadline": "2026-04-10T17:00:00Z"},
        validation_errors=[],
        step_number=0,
        task_id=task_id,
        task_description="",
        classified_reports={},
        routed_teams=[],
        drafted_messages=[],
        chosen_remediation=None,
        plan_published=False,
        errors_made=[],
        total_reward_so_far=0.0,
    )


def _example_rewards_single_triage() -> None:
    """
    Expected reward behavior:
    - Valid classification yields positive micro reward.
    - Choosing correct remediation for easy task adds an intermediate signal.
    - Publishing too early is penalized.
    """
    rc = RewardCalculator()
    s = _make_min_state("single_triage")
    # Note: this example keeps incident_reports empty to avoid requiring full IncidentReport objects.

    a1 = Action(action_type="classify_incident", parameters={"report_id": "r1", "severity": "high", "hazard_type": "choking"})
    r1 = rc.calculate_step_reward(a1, {"ok": True}, s)
    assert -0.5 <= r1 <= 1.0
    assert r1 > 0.0

    s.classified_reports["r1"] = {"severity": "high", "hazard_type": "choking"}
    a2 = Action(action_type="choose_remediation", parameters={"strategy": "service_bulletin"})
    r2 = rc.calculate_step_reward(a2, {"ok": True, "prerequisite_met": True}, s)
    assert r2 >= 0.10  # includes intermediate shaping for correct easy-task strategy

    a3 = Action(action_type="publish_plan", parameters={"plan_id": "p1"})
    r3 = rc.calculate_step_reward(a3, {"ok": True, "requirements_met": False}, s)
    assert r3 < 0.0  # premature publish should be negative


def _example_rewards_loop_penalty() -> None:
    """
    Demonstrates loop penalty escalation:
    repeating the same action signature multiple times should reduce step reward.
    """
    rc = RewardCalculator()
    s = _make_min_state("pattern_recall")

    act = Action(action_type="route", parameters={"team": "quality"})
    # First time: new team -> positive
    s.current_plan_state["action_history"] = []
    r1 = rc.calculate_step_reward(act, {"ok": True, "new_team": True}, s)

    # Pretend it happened already twice.
    s.current_plan_state["action_history"] = [
        {"signature": ("route", {"team": "quality"})},
        {"signature": ("route", {"team": "quality"})},
    ]
    r2 = rc.calculate_step_reward(act, {"ok": True, "new_team": False}, s)
    assert r2 < r1


def _example_final_reward_bounds() -> None:
    """
    Final episode score must always be in [0.0, 1.0].
    """
    rc = RewardCalculator()
    assert 0.0 <= rc.calculate_final_reward(False, steps_taken=5, errors=[]) <= 1.0
    assert 0.0 <= rc.calculate_final_reward(True, steps_taken=5, errors=[]) <= 1.0
    assert 0.0 <= rc.calculate_final_reward(True, steps_taken=25, errors=["bad"] * 100) <= 1.0


if __name__ == "__main__":
    _example_rewards_single_triage()
    _example_rewards_loop_penalty()
    _example_final_reward_bounds()
    print("RewardCalculator examples passed.")

