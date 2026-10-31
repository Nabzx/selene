#!/usr/bin/env bash
# Selene Complete Mission Pipeline & Human Sign-off
# Orchestrated by Gemini 4 Argon, certified by 1,000 Monte Carlo rollouts, signed off by human operator.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

# Load .env if present
if [ -f "$REPO_ROOT/.env" ]; then
  set -a
  source "$REPO_ROOT/.env"
  set +a
elif [ -f "$REPO_ROOT/apps/agent/.env" ]; then
  set -a
  source "$REPO_ROOT/apps/agent/.env"
  set +a
fi

USE_CACHE_FLAG=""
if [ -z "${GEMINI_API_KEY:-}" ]; then
  echo "INFO: GEMINI_API_KEY not detected. Using validated cached reasoning artifacts with live physics verifier."
  USE_CACHE_FLAG="--use-cache"
else
  echo "INFO: GEMINI_API_KEY detected. Running live Gemini 4 Argon orchestrator."
fi

echo "================================================================="
echo "               SELENE AUTONOMOUS MISSION PIPELINE                "
echo "  Target: Daedalus Crater Floor (Lunar Far Side, Lunar Night)   "
echo "================================================================="

# Phase 1: Mission Constraints -> Machine-Checkable Safety Spec
echo ""
echo "[Phase 1/5] Extracting machine-checkable safety specification..."
(cd apps/agent && uv run python -m selene_agent.orchestrator safety-spec \
  ../../docs/SUMMARY.md \
  -o ../../data/safety_spec.json \
  --cache ../../data/safety_spec.json \
  $USE_CACHE_FLAG)
echo "✓ Safety spec written to data/safety_spec.json"

# Phase 2: Real NASA DEM & Physics Simulation Environment Export
echo ""
echo "[Phase 2/5] Exporting NASA SLDEM2015 terrain crop & fleet environment..."
(cd apps/sim && uv run python scripts/export_demo_bundle.py \
  --checkpoint ../../data/checkpoints_nasa/mappo_stage3_curriculum.pt \
  --output-dir ../../data)
echo "✓ Environment bundle written to data/episodes.json"

# Phase 3: Fleet Trajectory Planning via Argon
echo ""
echo "[Phase 3/5] Argon orchestrator generating 4-hopper coordinated trajectory plan..."
(cd apps/agent && uv run python -m selene_agent.orchestrator plan \
  ../../data/episodes.json \
  -o ../../data/fleet_plan.json \
  --cache ../../data/fleet_plan.json \
  $USE_CACHE_FLAG)
echo "✓ Proposed fleet plan written to data/fleet_plan.json"

# Phase 4: Deterministic Monte Carlo Verification (1,000 Rollouts) & Revision
echo ""
echo "[Phase 4/5] Executing 1,000 Monte Carlo perturbed rollouts in Moon physics sim..."
(cd apps/agent && uv run python -m selene_agent.orchestrator revise \
  ../../data/fleet_plan.json \
  -o ../../data/fleet_plan_revised.json \
  --cache ../../data/fleet_plan_revised.json \
  $USE_CACHE_FLAG)
echo "✓ Verifier certified plan written to data/fleet_plan_revised.json"

# Phase 5: Log Telemetry & World Model Epistemic Uncertainty Review
echo ""
echo "[Phase 5/5] Diagnosing fleet telemetry & world model prediction hotspots..."
(cd apps/agent && uv run python -m selene_agent.orchestrator review-logs \
  ../../data/episodes.json \
  -o ../../data/log_review.json \
  --cache ../../data/log_review.json \
  $USE_CACHE_FLAG)
echo "✓ Telemetry review written to data/log_review.json"

# Phase 6: Human-in-the-Loop Uplink Sign-Off
echo ""
echo "================================================================="
echo "            HUMAN OPERATOR UPLINK CERTIFICATION & SIGN-OFF       "
echo "================================================================="
echo "The fleet plan has passed all automated simulator verifications."
echo "Orbiter relay contact window: OPEN at mission time t=10 s."
echo ""

(cd apps/agent && uv run python -m selene_agent.orchestrator signoff \
  ../../data/fleet_plan_revised.json \
  --safety-spec ../../data/safety_spec.json \
  --terrain ../../data/terrain/daedalus_nasa_sldem2015.npz \
  --mission-time 10 \
  --rollouts 1000 \
  -o ../../data/uplink_package.json)

echo ""
echo "================================================================="
echo "✓ MISSION UPLINK PACKAGE CREATED: data/uplink_package.json"
echo "================================================================="
