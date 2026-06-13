#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
RUN_STAGES=${RUN_STAGES:-"r0 r1 r2 r3-natural r3-common r4"}

for stage in ${RUN_STAGES}; do
  echo "============================================================"
  echo "Starting legacy e60 recovery stage: ${stage}"
  echo "============================================================"
  STAGE="${stage}" bash "${SCRIPT_DIR}/train_legacy_e60_recovery_stage_0614.sh"
done
