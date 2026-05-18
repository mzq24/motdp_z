#!/bin/bash
set -euo pipefail

if [[ -z "${WORKTREE_ROOT:-}" ]]; then
	SOURCE_PATH="${BASH_SOURCE[0]:-$0}"
	SCRIPT_DIR="$(cd "$(dirname "${SOURCE_PATH}")" && pwd)"
	WORKTREE_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
fi

MAX_SCENES=${MAX_SCENES:-8}
EXPERIMENT_NAME=${EXPERIMENT_NAME:-motdp_lead_navtest_smoke8}
GPU_ID=${GPU_ID:-2}

exec "${WORKTREE_ROOT}/scripts/_run_lead_navtest_full.sh"
