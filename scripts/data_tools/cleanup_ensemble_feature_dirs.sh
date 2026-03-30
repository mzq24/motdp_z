#!/bin/bash
# Remove intermediate ensemble feature directories after the averaged ensemble cache
# has been built.
#
# By default this script runs in dry-run mode and only prints the directories that
# would be removed. Use --apply to actually delete them.
#
# It targets only:
#   - transfuser_feature_m0
#   - transfuser_feature_m2
#
# It never touches the base transfuser_feature/ directories.
#
# Usage:
#   bash scripts/data_tools/cleanup_ensemble_feature_dirs.sh /path/to/pdm_lite
#   bash scripts/data_tools/cleanup_ensemble_feature_dirs.sh /path/to/pdm_lite --apply
#   bash scripts/data_tools/cleanup_ensemble_feature_dirs.sh /path/to/pdm_lite --apply --force

set -euo pipefail

usage() {
    cat <<'EOF'
Usage:
  bash scripts/data_tools/cleanup_ensemble_feature_dirs.sh <dataset_root> [--apply] [--force]

Arguments:
  <dataset_root>   Dataset root that contains route folders and tmp_data/

Options:
  --apply          Actually delete transfuser_feature_m0 and transfuser_feature_m2
  --force          Delete even if ensemble cache files are missing
  -h, --help       Show this help message

Behavior:
  - Default is dry-run only.
  - The script checks for these ensemble cache files under <dataset_root>/tmp_data:
      feature_index_ensemble.pkl
      bev_features_fp16_ensemble.bin
      bev_upsamples_fp16_ensemble.bin
  - If any are missing, deletion is blocked unless --force is provided.
EOF
}

if [[ $# -lt 1 ]]; then
    usage
    exit 1
fi

DATASET_ROOT=""
APPLY=0
FORCE=0

while [[ $# -gt 0 ]]; do
    case "$1" in
        --apply)
            APPLY=1
            shift
            ;;
        --force)
            FORCE=1
            shift
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        -*)
            echo "Unknown option: $1" >&2
            usage
            exit 1
            ;;
        *)
            if [[ -n "${DATASET_ROOT}" ]]; then
                echo "Unexpected extra positional argument: $1" >&2
                usage
                exit 1
            fi
            DATASET_ROOT="$1"
            shift
            ;;
    esac
done

if [[ -z "${DATASET_ROOT}" ]]; then
    echo "Dataset root is required." >&2
    usage
    exit 1
fi

if [[ ! -d "${DATASET_ROOT}" ]]; then
    echo "Dataset root does not exist: ${DATASET_ROOT}" >&2
    exit 1
fi

TMP_DATA_DIR="${DATASET_ROOT}/tmp_data"
REQUIRED_CACHE_FILES=(
    "${TMP_DATA_DIR}/feature_index_ensemble.pkl"
    "${TMP_DATA_DIR}/bev_features_fp16_ensemble.bin"
    "${TMP_DATA_DIR}/bev_upsamples_fp16_ensemble.bin"
)

missing_cache=()
for fp in "${REQUIRED_CACHE_FILES[@]}"; do
    if [[ ! -f "${fp}" ]]; then
        missing_cache+=("${fp}")
    fi
done

echo "=========================================="
echo "Cleanup Extra Ensemble Feature Directories"
echo "=========================================="
echo "Dataset root: ${DATASET_ROOT}"
echo "Mode: $([[ ${APPLY} -eq 1 ]] && echo APPLY || echo DRY-RUN)"
echo "Targets: transfuser_feature_m0, transfuser_feature_m2"
echo ""

if [[ ${#missing_cache[@]} -eq 0 ]]; then
    echo "Ensemble cache check: OK"
else
    echo "Ensemble cache check: MISSING"
    for fp in "${missing_cache[@]}"; do
        echo "  missing: ${fp}"
    done
    echo ""
    if [[ ${FORCE} -ne 1 ]]; then
        echo "Refusing to delete intermediate feature dirs because ensemble cache is incomplete."
        echo "If you really want to proceed anyway, rerun with --force."
        exit 1
    fi
    echo "Proceeding anyway because --force was provided."
fi

mapfile -t target_dirs < <(
    find "${DATASET_ROOT}" -type d \( -name 'transfuser_feature_m0' -o -name 'transfuser_feature_m2' \) | sort
)

if [[ ${#target_dirs[@]} -eq 0 ]]; then
    echo ""
    echo "No transfuser_feature_m0/m2 directories found."
    exit 0
fi

echo ""
echo "Found ${#target_dirs[@]} directories to remove:"
for dir in "${target_dirs[@]}"; do
    echo "  ${dir}"
done

if [[ ${APPLY} -ne 1 ]]; then
    echo ""
    echo "Dry-run only. No files were deleted."
    echo "Rerun with --apply to remove these directories."
    exit 0
fi

echo ""
echo "Deleting..."
for dir in "${target_dirs[@]}"; do
    rm -rf "${dir}"
    echo "  removed: ${dir}"
done

echo ""
echo "Done. Removed ${#target_dirs[@]} directories."
