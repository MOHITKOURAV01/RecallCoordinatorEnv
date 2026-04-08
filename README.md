---
title: RecallCoordinatorEnv
emoji: 🧠
colorFrom: blue
colorTo: indigo
sdk: docker
app_port: 7860
pinned: false
license: mit
---

# RecallCoordinatorEnv

![openenv](https://img.shields.io/badge/openenv%20validated-%E2%9C%85-blue)
![docker](https://img.shields.io/badge/Docker-ready-informational)
![hf](https://img.shields.io/badge/HF%20Spaces-%F0%9F%A4%97-yellow)
![python](https://img.shields.io/badge/Python-3.11%2B-blue)



## Overview

`RecallCoordinatorEnv` is a production-grade OpenEnv environment that simulates **consumer product safety recall coordination**—the work a real product safety officer does when incident reports begin to suggest a defect: triage, pattern detection, traceability checks, regulatory decisions, and cross-functional execution under **time and budget pressure**.

This domain matters for AI agent training because it combines **high-stakes decision-making** (safety and regulatory impact), **procedural compliance** (sequencing constraints and irreversible publish actions), and **artifact generation** (structured classifications + templated communications). It’s a realistic testbed for agents that must operate in messy, partially observed workflows without “game-like” shortcuts.

What makes this environment unique is the combination of (1) **deterministic synthetic incident streams**, (2) **traceability queries** (`query_db`) that change what a correct plan looks like, (3) a **multi-signal reward function** with loop detection and milestone shaping, and (4) **benchmark-style deterministic graders** with partial credit that are robust to malformed histories and state.

Target use cases include: training agents to follow regulated workflows, evaluating tool-using planning policies, benchmarking action sequencing and documentation quality, and stress-testing robustness under constraints (step budgets, deadlines, and spend).




## Environment Architecture :-

```
                 ┌──────────────────────────────────────────┐
                 │                Agent / Policy            │
                 │  (LLM via OpenAI client in inference.py) │
                 └───────────────────────────┬──────────────┘
                                             │ Action JSON
                                             v
┌─────────────────────────────────────────────────────────────────┐
│ FastAPI Server (`server/main.py`)                                │
│  POST /reset  ──> env.reset() ──> Observation JSON               │
│  POST /step   ──> env.step(action) ──> obs, reward, done, info   │
│  GET  /state  ──> env.state() ──> State JSON                     │
└─────────────────────────────────────────────────────────────────┘
                         │
                         v
┌─────────────────────────────────────────────────────────────────┐
│ RecallCoordinatorEnv (`server/env.py`)                            │
│  - applies validated actions                                      │
│  - updates State (classifications, routing, drafts, constraints)  │
│  - emits shaped step reward                                       │
│  - on done: runs deterministic grader -> info["grader_score"]     │
└─────────────────────────────────────────────────────────────────┘

State flow (single episode):
  reset() -> Observation(step=0) -> step(action) -> ... -> done -> final grader score
```




## Action Space

| action_name | parameters | description | example |
|---|---|---|---|
| `classify_incident` | `{report_id: str, severity: "low\|medium\|high\|critical", hazard_type: str}` | Assign a structured severity + hazard type to one report. `hazard_type` must be non-empty. | `{"action_type":"classify_incident","parameters":{"report_id":"r1","severity":"high","hazard_type":"choking"}}` |
| `route` | `{team: "legal\|quality\|comms\|ops"}` | Open/queue work with a team. Re-routing the same team yields reduced reward. | `{"action_type":"route","parameters":{"team":"quality"}}` |
| `query_db` | `{entity: str, filters: dict}` | Query deterministic synthetic “traceability DB”. Intended use: `entity="batch"` + `filters={"sku": "..."} or {"skus":[...]}`. | `{"action_type":"query_db","parameters":{"entity":"batch","filters":{"skus":["SPACE-HEATER-X","AIR-FRYER-Z"]}}}` |
| `draft_message` | `{channel: "customer\|regulator\|internal", template_id: str, variables: dict}` | Draft a templated message. Reward prefers drafts with all required variables filled. | `{"action_type":"draft_message","parameters":{"channel":"customer","template_id":"customer_notice_v1","variables":{"sku_list":"...","batch_list":"...","hazard_summary":"...","contact_info":"...","remediation_steps":"..."}}}` |
| `choose_remediation` | `{strategy: "repair\|replace\|refund\|service_bulletin\|recall"}` | Commit to a remediation strategy. Shaping prefers choosing after classification and discourages flip-flops. | `{"action_type":"choose_remediation","parameters":{"strategy":"recall"}}` |
| `publish_plan` | `{plan_id: str}` | Submit/publish the plan. Premature publish attempts are penalized; hard task requires routing+messages+query. | `{"action_type":"publish_plan","parameters":{"plan_id":"plan-001"}}` |

**Valid ranges and constraints**
- **`severity`**: must be one of `low|medium|high|critical`
- **`team`**: must be one of `legal|quality|comms|ops`
- **`channel`**: must be one of `customer|regulator|internal`
- **`template_id`**: one of `customer_notice_v1`, `regulator_notice_v1`, `internal_brief_v1`
- **Budget/deadline**: each step consumes “time” deterministically; actions spend budget deterministically; exceeding budget is penalized




## Observation Space

| field | type | description | example_value |
|---|---|---|---|
| `incident_reports` | `List[IncidentReport]` | The task’s incident stream (hardcoded synthetic data). | `[{"report_id":"r1","product_sku":"TOY-CHOK-001",...}]` |
| `current_plan_state` | `dict` | Operational artifacts: `tickets`, `drafts`, `approvals`, `queries`, `action_history`. | `{"tickets":[...],"queries":[...]}` |
| `constraints` | `dict` | Resource limits: `budget_remaining`, `deadline_hours`, `regulator_deadline`. | `{"budget_remaining":50000.0,"deadline_hours":48,...}` |
| `validation_errors` | `List[str]` | Errors from the *last* action validation (missing params, invalid enums). | `["Missing parameter: team"]` |
| `step_number` | `int` | Current step within the episode. | `7` |
| `task_id` | `str` | Task identifier. | `"pattern_recall"` |
| `task_description` | `str` | Natural-language objective. | `"Detect pattern, classify all..."` |

**What the agent can and cannot see**
- **Can see**: all incident report text/fields, current artifacts (tickets/drafts/queries), remaining budget and hours.
- **Cannot see**: any hidden “ground truth” causality; must infer action choices from patterns and constraints.
- **Deterministic**: same `task_id` always resets to the same initial observation.





## Task Descriptions

### Task 1 — `single_triage` (Easy): “Single report triage”
- **Difficulty**: Easy
- **Objective**: Correctly classify a toy choking hazard, route to Quality, and select a service bulletin remediation.
- **Success criteria**
  - Report `r1` severity classified as `high`
  - Routed to `quality`
  - Remediation set to `service_bulletin`
- **Grader scoring breakdown (0.0–1.0)**
  - +0.40: `r1.severity == "high"`
  - +0.30: `quality` routed
  - +0.30: remediation `service_bulletin`
- **Example perfect episode (step-by-step)**
  1. `classify_incident` for `r1` with `severity="high"`, `hazard_type="choking"`
  2. `route` team `"quality"`
  3. `choose_remediation` strategy `"service_bulletin"`
  4. `publish_plan` plan_id `"plan-001"`




### Task 2 — `pattern_recall` (Medium): “Cross-region pattern → recall”
- **Difficulty**: Medium
- **Objective**: Classify 5 related reports (same SKU across regions), mark injury cases as critical, draft customer + regulator messages, choose recall, and publish quickly.
- **Success criteria**
  - All 5 reports classified
  - Injury reports classified as `critical`
  - Customer + regulator drafts created
  - Remediation `recall`
  - Publish within 12 steps
  - No invalid actions
- **Grader scoring breakdown (0.0–1.0, capped)**
  - +0.20: all 5 classified
  - +0.10: injury reports are `critical`
  - +0.20: both message channels drafted (`customer` + `regulator`)
  - +0.20: remediation `recall`
  - +0.15: published within 12 steps
  - +0.15: no invalid actions
- **Example perfect episode (step-by-step)**
  1. Classify `r1..r5` (ensure injury ones are `critical`)
  2. `draft_message` channel `customer` with `customer_notice_v1` and filled variables
  3. `draft_message` channel `regulator` with `regulator_notice_v1` and filled variables
  4. `choose_remediation` strategy `"recall"`
  5. `publish_plan` plan_id `"plan-002"` (≤ 12 total steps)




### Task 3 — `full_recall_plan` (Hard): “Full recall plan under budget + deadline”
- **Difficulty**: Hard
- **Objective**: Execute a full recall plan across 12 reports and 2 SKUs: use traceability queries, coordinate all teams, draft all message types, choose recall, and publish without violating constraints.
- **Success criteria**
  - All 12 reports classified
  - Injury reports severity is `critical`
  - Used `query_db` to identify affected batches
  - Routed all 4 teams: `legal`, `quality`, `comms`, `ops`
  - Drafted 3 channels: `customer`, `regulator`, `internal`
  - Remediation is `recall`
  - Published while `deadline_hours > 0`
  - Budget not exceeded (`budget_remaining >= 0`)
- **Grader scoring breakdown (granular 0.0–1.0)**
  - +0.10: all 12 classified
  - +0.10: injury reports are `critical`
  - +0.10: `query_db` used for batch identification
  - +0.10: all 4 teams routed
  - +0.15: customer message with required fields (`sku_list`, `batch_list`, `hazard_summary`, `contact_info`)
  - +0.15: regulator message with required fields
  - +0.10: internal message drafted
  - +0.10: remediation `recall`
  - +0.05: published within deadline
  - +0.05: budget not exceeded
- **Example perfect episode (step-by-step)**
  1. `query_db` entity `"batch"` with `filters={"skus":["SPACE-HEATER-X","AIR-FRYER-Z"]}`
  2. Classify all `r1..r12` (injury ones as `critical`)
  3. `route` each team: `legal`, `quality`, `comms`, `ops`
  4. Draft messages:
     - `draft_message` `customer_notice_v1` (fill required variables)
     - `draft_message` `regulator_notice_v1` (fill required variables)
     - `draft_message` `internal_brief_v1`
  5. `choose_remediation` strategy `"recall"`
  6. `publish_plan` plan_id `"plan-003"` before budget/deadline run out




## Reward Function

The reward function is intentionally **trajectory-shaped**: it trains agents to (1) avoid invalid/looping behavior, (2) sequence work like a real recall process, and (3) generate complete artifacts before irreversible publish.

Step rewards are normalized to **`[-0.5, 1.0]`** and include micro-rewards, milestone shaping (coverage thresholds, required channels), and constraint pressure (budget/deadline). Final episode score remains in **`[0.0, 1.0]`**.

| situation | reward_signal | reasoning |
|---|---:|---|
| Valid `classify_incident` (useful fields) | + | Reinforces structured triage. |
| Increasing classification coverage (25%/75%/100%) | + | Milestone shaping for progress. |
| Routing a *new* team | + | Encourages cross-functional coordination. |
| Re-routing same team repeatedly | - | Discourages redundant loops. |
| `query_db` with SKU filters returning results | ++ | Rewards information-gathering that improves plan quality. |
| Drafting messages with all required variables filled | ++ | Rewards complete, “publish-ready” artifacts. |
| Choosing remediation *after* evidence (classifications) | + | Encourages correct sequencing. |
| Remediation flip-flop | -- | Models costly organizational churn. |
| Publishing with requirements met | +++ | Completion milestone. |
| Publishing prematurely (no classification / missing requirements) | -- | Irreversible bad action penalty. |
| Budget exceeded / deadline hits zero before publish | -- | Enforces real-world constraints. |





## Quick Start

### Docker setup (recommended)

```bash
docker build -t recall-coordinator-env .
docker run --rm -p 7860:7860 recall-coordinator-env
```

Sanity check:

```bash
curl -s http://localhost:7860/health
curl -s -X POST http://localhost:7860/reset -H "Content-Type: application/json" -d '{"task_id":"single_triage"}'
```

### Local setup (Python 3.11+)

```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
uvicorn server.main:app --host 0.0.0.0 --port 7860 --workers 1
```

### Run `inference.py`

```bash
export HF_TOKEN="YOUR_OPENAI_API_KEY"
export ENV_URL="http://localhost:7860"
# optional (defaults shown):
# export API_BASE_URL="https://api.openai.com/v1"
# export MODEL_NAME="gpt-4.1-mini"
python inference.py
```

`HF_TOKEN` is **required** by `inference.py` and is passed to the OpenAI-compatible client as the API key (despite the name).

### Hugging Face Spaces — secrets

- **FastAPI server only** (`uvicorn` in Docker): no API key needed for `/`, `/health`, `/docs`, `POST /reset`, `POST /step`, etc.
- **`inference.py`** (LLM agent that calls the env over HTTP): needs credentials and optional overrides.

In the Space: **Settings → Variables and secrets** (repository variables), add:

| Name | Required for inference | Default | Notes |
|------|------------------------|---------|--------|
| `HF_TOKEN` | Yes | — | OpenAI-compatible API key (same as local `export HF_TOKEN=...`). |
| `ENV_URL` | No | `http://localhost:7860` | Env API base URL. **Inside the same container:** `http://127.0.0.1:7860`. **From your laptop** hitting the public Space: use your Space app URL (see Space “Embed” / browser address). |
| `API_BASE_URL` | No | `https://api.openai.com/v1` | Compatible chat-completions API base. |
| `MODEL_NAME` | No | `gpt-4.1-mini` | Model id for the agent. |

If you run `inference.py` **locally** against a deployed Space, set `ENV_URL` to that Space’s HTTPS URL and `HF_TOKEN` to your API key; you do not have to add secrets to the Space unless you also run inference **inside** the Space container.

## Baseline Scores

Scores produced by running `inference.py` with `gpt-4.1-mini`
against the local server (`ENV_URL=http://localhost:7860`).

| Task | Difficulty | Model | Grader Score | Steps | Success |
|------|-----------|-------|-------------|-------|---------|
| single_triage | Easy | gpt-4.1-mini | 0.70 – 1.00 | 4–8 | ✅ |
| pattern_recall | Medium | gpt-4.1-mini | 0.50 – 0.85 | 8–14 | ⚠️ |
| full_recall_plan | Hard | gpt-4.1-mini | 0.30 – 0.65 | 15–20 | ⚠️ |

> Scores are reproducible within the stated range across 3 runs.
> Variance is due to LLM sampling temperature=0.2.
> Run `python inference.py` with env vars set to reproduce.

## OpenEnv Spec Compliance

- ✅ **Project structure** matches OpenEnv expectations (`server/` + root `inference.py`)
- ✅ **API** exposes `reset`, `step`, and `state` via FastAPI
- ✅ **Environment** implements `reset()`, `step()`, `state()` with clean episode management
- ✅ **Pydantic v2 strict models** for Action/Observation/State/RewardSignal
- ✅ **Exactly 3 tasks**: easy/medium/hard
- ✅ **Deterministic graders** return a score in `[0.0, 1.0]` with partial credit
- ✅ **Shaped rewards** (not binary), loop/invalid/time penalties, constraint pressure
- ✅ **Docker-ready** (`python:3.11-slim`, port `7860`) for HF Spaces
