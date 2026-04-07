#!/usr/bin/env bash
set -euo pipefail

IMAGE_NAME="${IMAGE_NAME:-recall-coordinator-env:test}"
CONTAINER_NAME="${CONTAINER_NAME:-recall-coordinator-env-test}"
PORT="${PORT:-7860}"

pass() { echo "PASS: $1"; }
fail() { echo "FAIL: $1"; exit 1; }

cleanup() {
  docker rm -f "${CONTAINER_NAME}" >/dev/null 2>&1 || true
}
trap cleanup EXIT

echo "Building image ${IMAGE_NAME}..."
docker build -t "${IMAGE_NAME}" .
pass "docker build"

echo "Starting container ${CONTAINER_NAME}..."
docker run -d --rm --name "${CONTAINER_NAME}" -p "${PORT}:7860" "${IMAGE_NAME}" >/dev/null

# Wait for health endpoint.
BASE_URL="http://localhost:${PORT}"
for i in {1..30}; do
  if curl -fsS "${BASE_URL}/health" >/dev/null 2>&1; then
    break
  fi
  sleep 1
done

HEALTH_JSON="$(curl -fsS "${BASE_URL}/health" || true)"
if [[ "${HEALTH_JSON}" == *'"status"'* ]]; then
  pass "/health reachable"
else
  echo "health response: ${HEALTH_JSON}"
  fail "/health reachable"
fi

RESET_JSON="$(curl -fsS -X POST "${BASE_URL}/reset" -H "Content-Type: application/json" -d '{"task_id":"single_triage"}' || true)"
if [[ "${RESET_JSON}" == *'"task_id":"single_triage"'* ]]; then
  pass "/reset returns observation"
else
  echo "reset response: ${RESET_JSON}"
  fail "/reset returns observation"
fi

STEP_JSON="$(curl -fsS -X POST "${BASE_URL}/step" -H "Content-Type: application/json" -d '{"action_type":"classify_incident","parameters":{"report_id":"r1","severity":"high","hazard_type":"choking"}}' || true)"
if [[ "${STEP_JSON}" == *'"reward"'* && "${STEP_JSON}" == *'"observation"'* ]]; then
  pass "/step returns (observation, reward, done, info)"
else
  echo "step response: ${STEP_JSON}"
  fail "/step returns (observation, reward, done, info)"
fi

pass "All checks passed"
