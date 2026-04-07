from __future__ import annotations

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, ConfigDict, Field


class IncidentReport(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    report_id: str
    product_sku: str
    batch_code: str
    hazard_description: str
    severity_raw: str
    date_reported: str
    region: str
    injury_reported: bool


class Action(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    action_type: str
    parameters: Dict[str, Any] = Field(default_factory=dict)


class Observation(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    incident_reports: List[IncidentReport]
    current_plan_state: Dict[str, Any]
    constraints: Dict[str, Any]
    validation_errors: List[str]
    step_number: int
    task_id: str
    task_description: str


class State(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    incident_reports: List[IncidentReport]
    current_plan_state: Dict[str, Any]
    constraints: Dict[str, Any]
    validation_errors: List[str]
    step_number: int
    task_id: str
    task_description: str

    classified_reports: Dict[str, Dict[str, Any]] = Field(default_factory=dict)
    routed_teams: List[str] = Field(default_factory=list)
    drafted_messages: List[Dict[str, Any]] = Field(default_factory=list)
    chosen_remediation: Optional[str] = None
    plan_published: bool = False
    errors_made: List[str] = Field(default_factory=list)
    total_reward_so_far: float = 0.0


class RewardSignal(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    step_reward: float
    cumulative_reward: float
    reward_reason: str
    done: bool

