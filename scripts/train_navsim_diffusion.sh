#!/usr/bin/env bash
set -euo pipefail

REPO_DIR="${REPO_DIR:-/workspace1/z_project/code/motdp_z_navsim_motdp}"
PYTHON="${PYTHON:-/workspace1/miniconda/envs/z_navsim_motdp/bin/python}"

GPU="${GPU:-3}"
RUN_NAME="${RUN_NAME:-navsim_simple_official_lr1e4_e60_b64}"
LOG_ROOT="${LOG_ROOT:-/workspace2/z_project/motdp_logs}"
LOG_DIR="${LOG_DIR:-${LOG_ROOT}/${RUN_NAME}}"

CACHE_DIR="${CACHE_DIR:-/workspace2/z_project/motdp_bev_cache_official_npy}"
FALLBACK_CACHE_DIR="${FALLBACK_CACHE_DIR:-/workspace2/z_project/motdp_bev_cache_official_final}"
TOKEN_FILTER_FILE="${TOKEN_FILTER_FILE:-}"
LOAD_MODE="${LOAD_MODE:-auto}"

BATCH_SIZE="${BATCH_SIZE:-64}"
EPOCHS="${EPOCHS:-60}"
LR="${LR:-1e-4}"
WEIGHT_DECAY="${WEIGHT_DECAY:-1e-4}"
WARMUP_EPOCHS="${WARMUP_EPOCHS:-3}"
LR_FINAL="${LR_FINAL:-1e-6}"
MAX_GRAD_NORM="${MAX_GRAD_NORM:-1.0}"
VAL_RATIO="${VAL_RATIO:-0.05}"
VAL_EVERY_EPOCHS="${VAL_EVERY_EPOCHS:-5}"
SAVE_EVERY_EPOCHS="${SAVE_EVERY_EPOCHS:-10}"
MAX_KEEP_CKPTS="${MAX_KEEP_CKPTS:-5}"
NUM_WORKERS="${NUM_WORKERS:-4}"
PREFETCH_FACTOR="${PREFETCH_FACTOR:-2}"
SEED="${SEED:-42}"
DEDUPE_TOKENS="${DEDUPE_TOKENS:-1}"

D_MODEL="${D_MODEL:-512}"
N_HEAD="${N_HEAD:-8}"
N_LAYER="${N_LAYER:-4}"
D_FFN="${D_FFN:-2048}"
P_DROP_ATTN="${P_DROP_ATTN:-0.1}"
P_DROP_EMB="${P_DROP_EMB:-0.1}"
NUM_INFERENCE_STEPS="${NUM_INFERENCE_STEPS:-10}"
NUM_TRAIN_TIMESTEPS="${NUM_TRAIN_TIMESTEPS:-1000}"
BETA_SCHEDULE="${BETA_SCHEDULE:-cosine}"
PREDICTION_TYPE="${PREDICTION_TYPE:-sample}"

USE_AMP="${USE_AMP:-1}"
AMP_DTYPE="${AMP_DTYPE:-bf16}"

mkdir -p "${LOG_DIR}"

args=(
  --cache-dir "${CACHE_DIR}"
  --fallback-cache-dir "${FALLBACK_CACHE_DIR}"
  --log-dir "${LOG_DIR}"
  --load-mode "${LOAD_MODE}"
  --batch-size "${BATCH_SIZE}"
  --epochs "${EPOCHS}"
  --lr "${LR}"
  --weight-decay "${WEIGHT_DECAY}"
  --warmup-epochs "${WARMUP_EPOCHS}"
  --lr-final "${LR_FINAL}"
  --max-grad-norm "${MAX_GRAD_NORM}"
  --val-ratio "${VAL_RATIO}"
  --val-every-epochs "${VAL_EVERY_EPOCHS}"
  --save-every-epochs "${SAVE_EVERY_EPOCHS}"
  --max-keep-ckpts "${MAX_KEEP_CKPTS}"
  --num-workers "${NUM_WORKERS}"
  --prefetch-factor "${PREFETCH_FACTOR}"
  --seed "${SEED}"
  --d-model "${D_MODEL}"
  --n-head "${N_HEAD}"
  --n-layer "${N_LAYER}"
  --d-ffn "${D_FFN}"
  --p-drop-attn "${P_DROP_ATTN}"
  --p-drop-emb "${P_DROP_EMB}"
  --num-inference-steps "${NUM_INFERENCE_STEPS}"
  --num-train-timesteps "${NUM_TRAIN_TIMESTEPS}"
  --beta-schedule "${BETA_SCHEDULE}"
  --prediction-type "${PREDICTION_TYPE}"
  --amp-dtype "${AMP_DTYPE}"
)

if [[ -n "${TOKEN_FILTER_FILE}" ]]; then
  args+=(--token-filter-file "${TOKEN_FILTER_FILE}")
fi
if [[ "${DEDUPE_TOKENS}" == "1" ]]; then
  args+=(--dedupe-tokens)
else
  args+=(--no-dedupe-tokens)
fi
if [[ "${USE_AMP}" == "1" ]]; then
  args+=(--use-amp)
else
  args+=(--no-use-amp)
fi
if [[ -n "${MAX_TRAIN_SAMPLES:-}" ]]; then
  args+=(--max-train-samples "${MAX_TRAIN_SAMPLES}")
fi
if [[ -n "${MAX_VAL_SAMPLES:-}" ]]; then
  args+=(--max-val-samples "${MAX_VAL_SAMPLES}")
fi

echo "========================================"
echo "NavSim cached-BEV diffusion training"
echo "  REPO_DIR:           ${REPO_DIR}"
echo "  GPU:                ${GPU}"
echo "  RUN_NAME:           ${RUN_NAME}"
echo "  LOG_DIR:            ${LOG_DIR}"
echo "  CACHE_DIR:          ${CACHE_DIR}"
echo "  FALLBACK_CACHE_DIR: ${FALLBACK_CACHE_DIR}"
echo "  TOKEN_FILTER_FILE:  ${TOKEN_FILTER_FILE:-<auto>}"
echo "  LOAD_MODE:          ${LOAD_MODE}"
echo "  BATCH_SIZE:         ${BATCH_SIZE}"
echo "  EPOCHS:             ${EPOCHS}"
echo "  LR:                 ${LR}"
echo "  VAL_EVERY_EPOCHS:   ${VAL_EVERY_EPOCHS}"
echo "  SAVE_EVERY_EPOCHS:  ${SAVE_EVERY_EPOCHS}"
echo "  AMP_DTYPE:          ${AMP_DTYPE}"
echo "========================================"

cd "${REPO_DIR}"
export CUDA_VISIBLE_DEVICES="${GPU}"
export PYTHONUNBUFFERED=1

"${PYTHON}" training/train_navsim_diffusion_cli.py "${args[@]}" "$@" \
  2>&1 | tee "${LOG_DIR}/console.log"
