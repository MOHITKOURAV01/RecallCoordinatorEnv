from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

from server.models import IncidentReport, State


@dataclass(frozen=True)
class TaskSpec:
    task_id: str
    difficulty: str
    description: str
    initial_reports: List[IncidentReport]
    initial_constraints: Dict[str, object]
    grader: Callable[[State, Sequence[Dict[str, Any]], Dict[str, Any]], float]


def _clamp01(x: float) -> float:
    """Clamp to strictly open interval (0, 1) as required by validator."""
    clamped = max(0.0, min(float(x), 1.0))
    if clamped <= 0.0:
        return 0.001
    if clamped >= 1.0:
        return 0.999
    return clamped


def _safe_list(x: Any) -> List[Any]:
    return x if isinstance(x, list) else []


def _safe_dict(x: Any) -> Dict[str, Any]:
    return x if isinstance(x, dict) else {}


def _action_type_counts(action_history: Sequence[Dict[str, Any]]) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    for a in action_history or []:
        if not isinstance(a, dict):
            continue
        sig = a.get("signature")
        if isinstance(sig, tuple) and len(sig) >= 1:
            at = str(sig[0])
        else:
            at = str(a.get("action_type", ""))
        if at:
            counts[at] = counts.get(at, 0) + 1
    return counts


def _channels_from_state(state: Any) -> set:
    drafted = getattr(state, "drafted_messages", None)
    channels = set()
    for m in _safe_list(drafted):
        if isinstance(m, dict) and m.get("channel"):
            channels.add(str(m["channel"]))
    return channels


def grade_task_1(final_state: State, action_history: Sequence[Dict[str, Any]], episode_info: Dict[str, Any]) -> float:
    """
    TASK 1 (EASY) — single_triage
    Objective: classify severity, route to quality, choose service_bulletin.

    Partial credit breakpoints (deterministic, 0–1):
    - +0.40 if report r1 classified as severity == "high"
    - +0.30 if routed to "quality"
    - +0.30 if chosen_remediation == "service_bulletin"
    Edge-case guardrails:
    - Any malformed/empty state returns 0.0 (never errors).
    """
    try:
        score = 0.0
        classified = _safe_dict(getattr(final_state, "classified_reports", {}))
        r1 = classified.get("r1", {}) if isinstance(classified, dict) else {}
        if isinstance(r1, dict) and str(r1.get("severity", "")).strip().lower() == "high":
            score += 0.4

        routed = getattr(final_state, "routed_teams", [])
        if isinstance(routed, list) and "quality" in routed:
            score += 0.3

        if str(getattr(final_state, "chosen_remediation", "") or "").strip().lower() == "service_bulletin":
            score += 0.3

        return _clamp01(score)
    except Exception:
        return 0.0


def grade_task_2(final_state: State, action_history: Sequence[Dict[str, Any]], episode_info: Dict[str, Any]) -> float:
    """
    TASK 2 (MEDIUM) — pattern_recall
    Objective: classify all 5, classify injury reports as critical, draft customer+regulator messages,
    choose recall, publish within 12 steps, avoid invalid actions.

    Partial credit (0–1, capped):
    - +0.20 if all 5 reports classified (any severities)
    - +0.10 if *injury* reports classified as "critical"
    - +0.20 if both messages drafted (customer + regulator)
    - +0.20 if remediation == "recall"
    - +0.15 if published within 12 steps
    - +0.15 if no invalid actions taken

    Edge cases:
    - Missing incident_reports/classified_reports/drafted_messages handled gracefully.
    - Uses action_history/episode_info only for defensive checks; no randomness.
    """
    try:
        score = 0.0
        reports = getattr(final_state, "incident_reports", [])
        reports_list: List[Any] = reports if isinstance(reports, list) else []
        report_ids = []
        injury_ids = []
        for r in reports_list:
            rid = getattr(r, "report_id", None)
            if isinstance(rid, str):
                report_ids.append(rid)
            if bool(getattr(r, "injury_reported", False)) and isinstance(rid, str):
                injury_ids.append(rid)

        classified = _safe_dict(getattr(final_state, "classified_reports", {}))
        if len(report_ids) == 5 and all(rid in classified for rid in report_ids):
            score += 0.2

        if injury_ids and all(
            str(_safe_dict(classified.get(rid, {})).get("severity", "")).strip().lower() == "critical" for rid in injury_ids
        ):
            score += 0.1

        channels = _channels_from_state(final_state)
        if "customer" in channels and "regulator" in channels:
            score += 0.2

        if str(getattr(final_state, "chosen_remediation", "") or "").strip().lower() == "recall":
            score += 0.2

        if bool(getattr(final_state, "plan_published", False)) and int(getattr(final_state, "step_number", 10**9)) <= 12:
            score += 0.15

        errors = getattr(final_state, "errors_made", [])
        if isinstance(errors, list) and len(errors) == 0:
            score += 0.15

        return _clamp01(score)
    except Exception:
        return 0.0


def grade_task_3(final_state: State, action_history: Sequence[Dict[str, Any]], episode_info: Dict[str, Any]) -> float:
    """
    TASK 3 (HARD) — full_recall_plan
    Objective: classify all 12, classify injury reports as critical, use query_db for batches,
    route all 4 teams, draft customer+regulator+internal, choose recall, publish within deadline+budget.

    Granular scoring (0–1, deterministic, partial credit):
    - +0.10 all 12 classified
    - +0.10 injury reports severity == critical
    - +0.10 used query_db to identify affected batches (via state.current_plan_state["queries"])
    - +0.10 routed all 4 teams (legal, quality, comms, ops)
    - +0.15 customer message drafted with required fields
    - +0.15 regulator message drafted with required fields
    - +0.10 internal message drafted
    - +0.10 remediation == recall
    - +0.05 published within deadline (deadline_hours > 0 at publish)
    - +0.05 budget not exceeded (budget_remaining >= 0)

    Edge cases: malformed fields -> 0.0 rather than exceptions.
    """
    try:
        score = 0.0
        reports = getattr(final_state, "incident_reports", [])
        reports_list: List[Any] = reports if isinstance(reports, list) else []
        report_ids = []
        injury_ids = []
        for r in reports_list:
            rid = getattr(r, "report_id", None)
            if isinstance(rid, str):
                report_ids.append(rid)
            if bool(getattr(r, "injury_reported", False)) and isinstance(rid, str):
                injury_ids.append(rid)

        classified = _safe_dict(getattr(final_state, "classified_reports", {}))
        if len(report_ids) == 12 and all(rid in classified for rid in report_ids):
            score += 0.10

        if injury_ids and all(
            str(_safe_dict(classified.get(rid, {})).get("severity", "")).strip().lower() == "critical" for rid in injury_ids
        ):
            score += 0.10

        cps = _safe_dict(getattr(final_state, "current_plan_state", {}))
        queries = cps.get("queries", [])
        if isinstance(queries, list) and any(isinstance(q, dict) and str(q.get("entity", "")).lower() in {"batch", "batches", "production"} for q in queries):
            score += 0.10

        routed = getattr(final_state, "routed_teams", [])
        if isinstance(routed, list) and all(t in routed for t in ["legal", "quality", "comms", "ops"]):
            score += 0.10

        msgs = [m for m in _safe_list(getattr(final_state, "drafted_messages", [])) if isinstance(m, dict)]
        msg_by_channel = {str(m.get("channel")): m for m in msgs if m.get("channel")}

        def _has_required_fields(m: Dict[str, Any]) -> bool:
            vars_ = m.get("variables", {})
            if not isinstance(vars_, dict):
                return False
            required = ["sku_list", "batch_list", "hazard_summary", "contact_info"]
            return all(str(vars_.get(k, "")).strip() for k in required)

        if "customer" in msg_by_channel and _has_required_fields(msg_by_channel["customer"]):
            score += 0.15
        if "regulator" in msg_by_channel and _has_required_fields(msg_by_channel["regulator"]):
            score += 0.15
        if "internal" in msg_by_channel:
            score += 0.10

        if str(getattr(final_state, "chosen_remediation", "") or "").strip().lower() == "recall":
            score += 0.10

        constraints = _safe_dict(getattr(final_state, "constraints", {}))
        deadline_hours = constraints.get("deadline_hours")
        if bool(getattr(final_state, "plan_published", False)) and isinstance(deadline_hours, int) and deadline_hours > 0:
            score += 0.05

        budget_remaining = constraints.get("budget_remaining")
        if isinstance(budget_remaining, (int, float)) and float(budget_remaining) >= 0.0:
            score += 0.05

        return _clamp01(score)
    except Exception:
        return 0.0


def grade_task_1_state(state: State) -> float:
    """Back-compat wrapper: grade using state only."""
    hist = _safe_list(_safe_dict(getattr(state, "current_plan_state", {})).get("action_history"))
    return grade_task_1(state, hist, {})


def grade_task_2_state(state: State) -> float:
    """Back-compat wrapper: grade using state only."""
    hist = _safe_list(_safe_dict(getattr(state, "current_plan_state", {})).get("action_history"))
    return grade_task_2(state, hist, {})


def grade_task_3_state(state: State) -> float:
    """Back-compat wrapper: grade using state only."""
    hist = _safe_list(_safe_dict(getattr(state, "current_plan_state", {})).get("action_history"))
    return grade_task_3(state, hist, {})


def grade_task_4(
    final_state: State,
    action_history: Sequence[Dict[str, Any]],
    episode_info: Dict[str, Any],
) -> float:
    """
    TASK 4 (EXPERT) — regulatory_audit
    Scenario: CPSC audit letter received. Agent must:
    - Classify all 8 incident reports
    - Query DB for all affected batches
    - Route to legal + quality + comms
    - Draft regulator response + internal corrective action plan
    - Choose remediation strategy
    - Publish within budget and deadline

    Scoring (0.0-1.0, deterministic, partial credit):
    +0.10 all 8 reports classified
    +0.10 injury reports classified as critical
    +0.10 query_db used for batch identification
    +0.10 legal + quality + comms routed (3 teams)
    +0.15 regulator message drafted with required fields
    +0.15 internal message drafted
    +0.10 remediation chosen
    +0.10 plan published
    +0.05 within deadline
    +0.05 within budget
    """
    try:
        score = 0.0
        reports = getattr(final_state, "incident_reports", [])
        reports_list = reports if isinstance(reports, list) else []
        report_ids = []
        injury_ids = []
        for r in reports_list:
            rid = getattr(r, "report_id", None)
            if isinstance(rid, str):
                report_ids.append(rid)
            if bool(getattr(r, "injury_reported", False)) and isinstance(rid, str):
                injury_ids.append(rid)

        classified = _safe_dict(getattr(final_state, "classified_reports", {}))

        # +0.10 all 8 classified
        if len(report_ids) == 8 and all(rid in classified for rid in report_ids):
            score += 0.10

        # +0.10 injury reports = critical
        if injury_ids and all(
            str(_safe_dict(classified.get(rid, {})).get("severity", "")).strip().lower() == "critical"
            for rid in injury_ids
        ):
            score += 0.10

        # +0.10 query_db used
        cps = _safe_dict(getattr(final_state, "current_plan_state", {}))
        queries = cps.get("queries", [])
        if isinstance(queries, list) and any(
            isinstance(q, dict) and str(q.get("entity", "")).lower() in {"batch", "batches", "production"}
            for q in queries
        ):
            score += 0.10

        # +0.10 legal + quality + comms routed
        routed = getattr(final_state, "routed_teams", [])
        if isinstance(routed, list) and all(t in routed for t in ["legal", "quality", "comms"]):
            score += 0.10

        # +0.15 regulator message with required fields
        msgs = [m for m in _safe_list(getattr(final_state, "drafted_messages", [])) if isinstance(m, dict)]
        msg_by_channel = {str(m.get("channel")): m for m in msgs if m.get("channel")}

        def _has_core_fields(m: Dict[str, Any]) -> bool:
            vars_ = m.get("variables", {})
            if not isinstance(vars_, dict):
                return False
            return all(str(vars_.get(k, "")).strip() for k in ["sku_list", "batch_list", "hazard_summary", "contact_info"])

        if "regulator" in msg_by_channel and _has_core_fields(msg_by_channel["regulator"]):
            score += 0.15

        # +0.15 internal message drafted
        if "internal" in msg_by_channel:
            score += 0.15

        # +0.10 remediation chosen
        if str(getattr(final_state, "chosen_remediation", "") or "").strip().lower() in {
            "recall", "repair", "replace", "refund", "service_bulletin"
        }:
            score += 0.10

        # +0.10 plan published
        if bool(getattr(final_state, "plan_published", False)):
            score += 0.10

        # +0.05 within deadline
        constraints = _safe_dict(getattr(final_state, "constraints", {}))
        deadline_hours = constraints.get("deadline_hours")
        if isinstance(deadline_hours, int) and deadline_hours > 0:
            score += 0.05

        # +0.05 within budget
        budget_remaining = constraints.get("budget_remaining")
        if isinstance(budget_remaining, (int, float)) and float(budget_remaining) >= 0.0:
            score += 0.05

        return _clamp01(score)
    except Exception:
        return 0.0


def grade_task_4_state(state: State) -> float:
    hist = _safe_list(_safe_dict(getattr(state, "current_plan_state", {})).get("action_history"))
    return grade_task_4(state, hist, {})


class GraderValidator:
    """
    Validates grader contracts:
    - score in [0,1]
    - determinism (same input twice -> same score)
    - prints a concise report
    """

    def run(self) -> None:
        report: List[str] = []

        def _check(name: str, fn: Callable[[State, Sequence[Dict[str, Any]], Dict[str, Any]], float], st: State) -> None:
            hist = _safe_list(_safe_dict(st.current_plan_state).get("action_history"))
            info = {"steps_taken": st.step_number}
            s1 = fn(st, hist, info)
            s2 = fn(st, hist, info)
            ok_bounds = 0.0 <= s1 <= 1.0 and 0.0 <= s2 <= 1.0
            ok_det = abs(s1 - s2) < 1e-12
            report.append(f"- {name}: score={s1:.3f} bounds={'OK' if ok_bounds else 'FAIL'} determinism={'OK' if ok_det else 'FAIL'}")

        # Build synthetic states using existing task initial states and then mutate.
        base1 = State(
            incident_reports=TASKS["single_triage"].initial_reports,
            current_plan_state={"action_history": []},
            constraints=dict(TASKS["single_triage"].initial_constraints),
            validation_errors=[],
            step_number=5,
            task_id="single_triage",
            task_description="",
            classified_reports={"r1": {"severity": "high", "hazard_type": "choking"}},
            routed_teams=["quality"],
            drafted_messages=[],
            chosen_remediation="service_bulletin",
            plan_published=True,
            errors_made=[],
            total_reward_so_far=0.0,
        )

        base2 = State(
            incident_reports=TASKS["pattern_recall"].initial_reports,
            current_plan_state={"action_history": []},
            constraints=dict(TASKS["pattern_recall"].initial_constraints),
            validation_errors=[],
            step_number=10,
            task_id="pattern_recall",
            task_description="",
            classified_reports={r.report_id: {"severity": "high", "hazard_type": "electrical"} for r in TASKS["pattern_recall"].initial_reports},
            routed_teams=["quality", "comms"],
            drafted_messages=[{"channel": "customer", "template_id": "customer_notice_v1", "variables": {}}, {"channel": "regulator", "template_id": "regulator_notice_v1", "variables": {}}],
            chosen_remediation="recall",
            plan_published=True,
            errors_made=[],
            total_reward_so_far=0.0,
        )
        # injury reports critical
        for r in TASKS["pattern_recall"].initial_reports:
            if r.injury_reported:
                base2.classified_reports[r.report_id]["severity"] = "critical"

        base3 = State(
            incident_reports=TASKS["full_recall_plan"].initial_reports,
            current_plan_state={"action_history": [], "queries": [{"entity": "batch", "filters": {"skus": ["SPACE-HEATER-X"]}}]},
            constraints=dict(TASKS["full_recall_plan"].initial_constraints),
            validation_errors=[],
            step_number=14,
            task_id="full_recall_plan",
            task_description="",
            classified_reports={r.report_id: {"severity": ("critical" if r.injury_reported else "high"), "hazard_type": "fire"} for r in TASKS["full_recall_plan"].initial_reports},
            routed_teams=["legal", "quality", "comms", "ops"],
            drafted_messages=[
                {"channel": "customer", "template_id": "customer_notice_v1", "variables": {"sku_list": "A", "batch_list": "B", "hazard_summary": "C", "contact_info": "D"}},
                {"channel": "regulator", "template_id": "regulator_notice_v1", "variables": {"sku_list": "A", "batch_list": "B", "hazard_summary": "C", "contact_info": "D"}},
                {"channel": "internal", "template_id": "internal_brief_v1", "variables": {}},
            ],
            chosen_remediation="recall",
            plan_published=True,
            errors_made=[],
            total_reward_so_far=0.0,
        )

        _check("grade_task_1", grade_task_1, base1)
        _check("grade_task_2", grade_task_2, base2)
        _check("grade_task_3", grade_task_3, base3)

        print("GraderValidator report:")
        for line in report:
            print(line)


def _sample_tests() -> None:
    """
    Sample tests required by benchmark spec:
    For each task: perfect (1.0), partial (~0.5), zero (0.0)
    """
    # ---- Task 1
    t1 = TASKS["single_triage"]
    s1_perfect = State(
        incident_reports=t1.initial_reports,
        current_plan_state={"action_history": []},
        constraints=dict(t1.initial_constraints),
        validation_errors=[],
        step_number=4,
        task_id="single_triage",
        task_description="",
        classified_reports={"r1": {"severity": "high", "hazard_type": "choking"}},
        routed_teams=["quality"],
        drafted_messages=[],
        chosen_remediation="service_bulletin",
        plan_published=True,
        errors_made=[],
        total_reward_so_far=0.0,
    )
    assert 0.9 <= grade_task_1(s1_perfect, [], {}) < 1.0

    s1_partial = State.model_validate(s1_perfect.model_dump(mode="python"))
    s1_partial.routed_teams = []
    assert 0.35 <= grade_task_1(s1_partial, [], {}) <= 0.75

    s1_zero = State.model_validate(s1_perfect.model_dump(mode="python"))
    s1_zero.classified_reports = {}
    s1_zero.routed_teams = []
    s1_zero.chosen_remediation = None
    assert grade_task_1(s1_zero, [], {}) <= 0.01

    # ---- Task 2
    t2 = TASKS["pattern_recall"]
    s2_perfect = State(
        incident_reports=t2.initial_reports,
        current_plan_state={"action_history": []},
        constraints=dict(t2.initial_constraints),
        validation_errors=[],
        step_number=10,
        task_id="pattern_recall",
        task_description="",
        classified_reports={r.report_id: {"severity": ("critical" if r.injury_reported else "high"), "hazard_type": "electrical"} for r in t2.initial_reports},
        routed_teams=["quality", "comms"],
        drafted_messages=[{"channel": "customer"}, {"channel": "regulator"}],
        chosen_remediation="recall",
        plan_published=True,
        errors_made=[],
        total_reward_so_far=0.0,
    )
    assert 0.9 <= grade_task_2(s2_perfect, [], {}) < 1.0

    s2_partial = State.model_validate(s2_perfect.model_dump(mode="python"))
    s2_partial.plan_published = False
    s2_partial.drafted_messages = [{"channel": "customer"}]
    # Expect mid score (classified + injuries + remediation but missing publish/regulator msg)
    ps = grade_task_2(s2_partial, [], {})
    assert 0.35 <= ps <= 0.75

    s2_zero = State.model_validate(s2_perfect.model_dump(mode="python"))
    s2_zero.classified_reports = {}
    s2_zero.drafted_messages = []
    s2_zero.chosen_remediation = None
    s2_zero.plan_published = False
    s2_zero.errors_made = ["invalid"]
    assert grade_task_2(s2_zero, [], {}) <= 0.01

    # ---- Task 3
    t3 = TASKS["full_recall_plan"]
    s3_perfect = State(
        incident_reports=t3.initial_reports,
        current_plan_state={"action_history": [], "queries": [{"entity": "batch", "filters": {"skus": ["AIR-FRYER-Z"]}}]},
        constraints=dict(t3.initial_constraints),
        validation_errors=[],
        step_number=14,
        task_id="full_recall_plan",
        task_description="",
        classified_reports={r.report_id: {"severity": ("critical" if r.injury_reported else "high"), "hazard_type": "fire"} for r in t3.initial_reports},
        routed_teams=["legal", "quality", "comms", "ops"],
        drafted_messages=[
            {"channel": "customer", "variables": {"sku_list": "A", "batch_list": "B", "hazard_summary": "C", "contact_info": "D"}},
            {"channel": "regulator", "variables": {"sku_list": "A", "batch_list": "B", "hazard_summary": "C", "contact_info": "D"}},
            {"channel": "internal"},
        ],
        chosen_remediation="recall",
        plan_published=True,
        errors_made=[],
        total_reward_so_far=0.0,
    )
    assert 0.9 <= grade_task_3(s3_perfect, [], {}) < 1.0

    s3_partial = State.model_validate(s3_perfect.model_dump(mode="python"))
    s3_partial.drafted_messages = [{"channel": "customer", "variables": {"sku_list": "A", "batch_list": "B", "hazard_summary": "C", "contact_info": "D"}}]
    s3_partial.routed_teams = ["quality"]
    s3_partial.current_plan_state = {"action_history": []}  # remove query evidence
    ps3 = grade_task_3(s3_partial, [], {})
    assert 0.25 <= ps3 <= 0.70

    s3_zero = State.model_validate(s3_perfect.model_dump(mode="python"))
    s3_zero.classified_reports = {}
    s3_zero.routed_teams = []
    s3_zero.drafted_messages = []
    s3_zero.chosen_remediation = None
    s3_zero.plan_published = False
    s3_zero.constraints["budget_remaining"] = -1.0
    s3_zero.current_plan_state = {"action_history": []}
    assert grade_task_3(s3_zero, [], {}) <= 0.01


TASKS: Dict[str, TaskSpec] = {
    "single_triage": TaskSpec(
        task_id="single_triage",
        difficulty="easy",
        description="Correctly classify severity, route to quality team, choose remediation = service_bulletin.",
        initial_reports=[
            IncidentReport(
                report_id="r1",
                product_sku="TOY-CHOK-001",
                batch_code="BATCH-A1",
                hazard_description="Small detachable part may pose choking hazard for children under 3.",
                severity_raw="customer complaint: near-choking event; no injury reported",
                date_reported="2026-04-01",
                region="NA",
                injury_reported=False,
            )
        ],
        initial_constraints={"budget_remaining": 10000.0, "deadline_hours": 72, "regulator_deadline": "2026-04-10T17:00:00Z"},
        grader=grade_task_1,
    ),
    "pattern_recall": TaskSpec(
        task_id="pattern_recall",
        difficulty="medium",
        description="Detect pattern, classify all reports, draft customer + regulator messages, choose remediation = recall, publish quickly.",
        initial_reports=[
            IncidentReport(
                report_id="r1",
                product_sku="KITCH-MIX-200",
                batch_code="KM200-24-11",
                hazard_description="Mixer emits smoke; potential electrical short.",
                severity_raw="smoke observed; unit shut off; minor burn reported",
                date_reported="2026-03-28",
                region="NA",
                injury_reported=True,
            ),
            IncidentReport(
                report_id="r2",
                product_sku="KITCH-MIX-200",
                batch_code="KM200-24-11",
                hazard_description="Burning smell and sparks from base after 10 minutes.",
                severity_raw="customer unplugged; no injury",
                date_reported="2026-03-29",
                region="EU",
                injury_reported=False,
            ),
            IncidentReport(
                report_id="r3",
                product_sku="KITCH-MIX-200",
                batch_code="KM200-24-11",
                hazard_description="Electrical short; tripped breaker; smoke.",
                severity_raw="treated for minor smoke inhalation",
                date_reported="2026-03-30",
                region="APAC",
                injury_reported=True,
            ),
            IncidentReport(
                report_id="r4",
                product_sku="KITCH-MIX-200",
                batch_code="KM200-24-11",
                hazard_description="Overheats; casing hot to touch; smell of burning plastic.",
                severity_raw="no injury; stopped use",
                date_reported="2026-03-30",
                region="NA",
                injury_reported=False,
            ),
            IncidentReport(
                report_id="r5",
                product_sku="KITCH-MIX-200",
                batch_code="KM200-24-11",
                hazard_description="Sparks visible; scorch marks near power cord entry.",
                severity_raw="no injury; property damage to countertop",
                date_reported="2026-03-31",
                region="EU",
                injury_reported=False,
            ),
        ],
        initial_constraints={"budget_remaining": 25000.0, "deadline_hours": 96, "regulator_deadline": "2026-04-08T17:00:00Z"},
        grader=grade_task_2,
    ),
    "full_recall_plan": TaskSpec(
        task_id="full_recall_plan",
        difficulty="hard",
        description=(
            "Identify affected SKUs/batches via query_db, classify all, route to legal/quality/comms/ops, "
            "draft customer+regulator+internal messages, choose recall, publish within budget/deadline."
        ),
        initial_reports=[
            IncidentReport(
                report_id="r1",
                product_sku="SPACE-HEATER-X",
                batch_code="SHX-25-02",
                hazard_description="Heater emits sparks; potential fire hazard.",
                severity_raw="burned carpet; no injury",
                date_reported="2026-03-20",
                region="NA",
                injury_reported=False,
            ),
            IncidentReport(
                report_id="r2",
                product_sku="SPACE-HEATER-X",
                batch_code="SHX-25-02",
                hazard_description="Overheating and smoke; unit failure.",
                severity_raw="minor burn on hand",
                date_reported="2026-03-21",
                region="EU",
                injury_reported=True,
            ),
            IncidentReport(
                report_id="r3",
                product_sku="SPACE-HEATER-X",
                batch_code="SHX-25-03",
                hazard_description="Crackling sound then sparks near plug.",
                severity_raw="no injury; outlet damaged",
                date_reported="2026-03-22",
                region="NA",
                injury_reported=False,
            ),
            IncidentReport(
                report_id="r4",
                product_sku="SPACE-HEATER-X",
                batch_code="SHX-25-02",
                hazard_description="Small fire started inside casing.",
                severity_raw="treated for smoke inhalation",
                date_reported="2026-03-23",
                region="APAC",
                injury_reported=True,
            ),
            IncidentReport(
                report_id="r5",
                product_sku="AIR-FRYER-Z",
                batch_code="AFZ-24-12",
                hazard_description="Basket latch fails; hot contents spilled.",
                severity_raw="second-degree burn reported",
                date_reported="2026-03-24",
                region="NA",
                injury_reported=True,
            ),
            IncidentReport(
                report_id="r6",
                product_sku="AIR-FRYER-Z",
                batch_code="AFZ-24-12",
                hazard_description="Latch failure; basket detached during removal.",
                severity_raw="no injury; near miss",
                date_reported="2026-03-25",
                region="EU",
                injury_reported=False,
            ),
            IncidentReport(
                report_id="r7",
                product_sku="AIR-FRYER-Z",
                batch_code="AFZ-24-11",
                hazard_description="Handle loosens; basket tips unexpectedly.",
                severity_raw="minor burn on wrist",
                date_reported="2026-03-26",
                region="APAC",
                injury_reported=True,
            ),
            IncidentReport(
                report_id="r8",
                product_sku="SPACE-HEATER-X",
                batch_code="SHX-25-02",
                hazard_description="Sparks and smoke after 5 minutes.",
                severity_raw="no injury",
                date_reported="2026-03-27",
                region="EU",
                injury_reported=False,
            ),
            IncidentReport(
                report_id="r9",
                product_sku="SPACE-HEATER-X",
                batch_code="SHX-25-03",
                hazard_description="Cord melts; electrical short.",
                severity_raw="no injury; breaker tripped",
                date_reported="2026-03-28",
                region="NA",
                injury_reported=False,
            ),
            IncidentReport(
                report_id="r10",
                product_sku="AIR-FRYER-Z",
                batch_code="AFZ-24-12",
                hazard_description="Latch breaks; hot oil spilled.",
                severity_raw="first-degree burn",
                date_reported="2026-03-29",
                region="EU",
                injury_reported=True,
            ),
            IncidentReport(
                report_id="r11",
                product_sku="AIR-FRYER-Z",
                batch_code="AFZ-24-11",
                hazard_description="Basket detaches; contents spilled.",
                severity_raw="no injury; property damage",
                date_reported="2026-03-30",
                region="NA",
                injury_reported=False,
            ),
            IncidentReport(
                report_id="r12",
                product_sku="SPACE-HEATER-X",
                batch_code="SHX-25-02",
                hazard_description="Internal arcing; smoke; small flame.",
                severity_raw="minor burn reported",
                date_reported="2026-03-31",
                region="APAC",
                injury_reported=True,
            ),
        ],
        initial_constraints={"budget_remaining": 50000.0, "deadline_hours": 48, "regulator_deadline": "2026-04-02T17:00:00Z"},
        grader=grade_task_3,
    ),
    "regulatory_audit": TaskSpec(
        task_id="regulatory_audit",
        difficulty="expert",
        description=(
            "CPSC audit received. Classify all reports, query affected batches, "
            "route to legal/quality/comms, draft regulator response + internal plan, "
            "choose remediation, publish within budget and deadline."
        ),
        initial_reports=[
            IncidentReport(
                report_id="r1",
                product_sku="SPACE-HEATER-X",
                batch_code="SHX-25-02",
                hazard_description="Heater sparks; fire risk confirmed by lab.",
                severity_raw="Property damage; no injury",
                date_reported="2026-01-15",
                region="NA",
                injury_reported=False,
            ),
            IncidentReport(
                report_id="r2",
                product_sku="SPACE-HEATER-X",
                batch_code="SHX-25-02",
                hazard_description="Overheating causes smoke and burn marks.",
                severity_raw="Minor burn on hand reported",
                date_reported="2026-01-18",
                region="EU",
                injury_reported=True,
            ),
            IncidentReport(
                report_id="r3",
                product_sku="AIR-FRYER-Z",
                batch_code="AFZ-24-12",
                hazard_description="Basket latch fails; hot contents spilled.",
                severity_raw="Second-degree burn on forearm",
                date_reported="2026-01-20",
                region="NA",
                injury_reported=True,
            ),
            IncidentReport(
                report_id="r4",
                product_sku="AIR-FRYER-Z",
                batch_code="AFZ-24-12",
                hazard_description="Latch mechanism defective; near miss.",
                severity_raw="No injury; property damage",
                date_reported="2026-01-22",
                region="APAC",
                injury_reported=False,
            ),
            IncidentReport(
                report_id="r5",
                product_sku="SPACE-HEATER-X",
                batch_code="SHX-25-03",
                hazard_description="Electrical arcing; tripped breaker.",
                severity_raw="No injury; outlet damaged",
                date_reported="2026-01-25",
                region="NA",
                injury_reported=False,
            ),
            IncidentReport(
                report_id="r6",
                product_sku="KITCH-MIX-200",
                batch_code="KM200-24-11",
                hazard_description="Smoke from motor; electrical short.",
                severity_raw="Smoke inhalation; treated at ER",
                date_reported="2026-01-28",
                region="EU",
                injury_reported=True,
            ),
            IncidentReport(
                report_id="r7",
                product_sku="KITCH-MIX-200",
                batch_code="KM200-24-11",
                hazard_description="Sparks from base during normal use.",
                severity_raw="No injury; unit destroyed",
                date_reported="2026-01-30",
                region="NA",
                injury_reported=False,
            ),
            IncidentReport(
                report_id="r8",
                product_sku="AIR-FRYER-Z",
                batch_code="AFZ-24-11",
                hazard_description="Handle loosens; basket tips; hot oil spill.",
                severity_raw="First-degree burn reported",
                date_reported="2026-02-01",
                region="EU",
                injury_reported=True,
            ),
        ],
        initial_constraints={
            "budget_remaining": 75000.0,
            "deadline_hours": 48,
            "regulator_deadline": "2026-04-12T17:00:00Z",
        },
        grader=grade_task_4,
    ),
}


if __name__ == "__main__":
    _sample_tests()
    GraderValidator().run()

