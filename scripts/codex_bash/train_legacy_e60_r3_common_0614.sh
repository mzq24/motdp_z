#!/usr/bin/env bash
set -euo pipefail
STAGE=r3-common MASTER_PORT=${MASTER_PORT:-29624} bash "$(dirname "$0")/train_legacy_e60_recovery_stage_0614.sh"
