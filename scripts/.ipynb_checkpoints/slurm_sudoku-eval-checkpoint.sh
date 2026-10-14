#!/bin/bash

#SBATCH --job-name=sudoku-eval
#SBATCH --output=/projects/u6tp/tw1320/submit/out/sudoku_eval_%A_%a.out
#SBATCH --error=/projects/u6tp/tw1320/submit/out/sudoku_eval_%A_%a.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=2
#SBATCH --time=24:00:00

set -euo pipefail

# sbatch exports the submitting shell's environment, and an interactive session's
# TMPDIR may point somewhere the compute node cannot create (torch builds its
# inductor cache under it at import). Pin the job to node-local scratch.
export TMPDIR="${SLURM_TMPDIR:-/tmp}"

echo "==============================================================================================="
echo "Job ID: ${SLURM_JOB_ID:-local}"
echo "Array job ID: ${SLURM_ARRAY_JOB_ID:-none}"
echo "Array task ID: ${SLURM_ARRAY_TASK_ID:-0}"
echo "Host: $(hostname)"
echo "Started: $(date)"
echo "==============================================================================================="

# ---------------------------------------------------------------------------
# Persistent locations (scratch is wiped, so checkpoints, data and the venv live outside it).
# ---------------------------------------------------------------------------
persist_root="/projects/u6tp/tw1320/diffgemma"
data_root="${persist_root}/sudoku_data"
pretrained_root="${persist_root}/gemma_weights"
trained_root="${persist_root}/trained"

# Repo checkout. scripts/isambard_cuda13.sh activates its .venv inside the CUDA 13
# container, so the venv is not activated on the host.
repo_dir="${HOME}/gemma"

# The launcher runs apptainer with --cleanenv; forward what the run needs.
export APPTAINERENV_GEMMA_CACHE_DIR="${persist_root}/gemma_cache"
# wandb falls back to ~/.netrc (home is bound) if no key is set.
if [[ -v WANDB_API_KEY ]]; then
    export APPTAINERENV_WANDB_API_KEY="${WANDB_API_KEY}"
fi

# ---------------------------------------------------------------------------
# Config set up
# ---------------------------------------------------------------------------

# Reuse a training run, not this evaluation job's ID. Override these when
# evaluating another run or checkpoint, e.g. SUDOKU_EVAL_STEP=2000 sbatch ...
workdir="${SUDOKU_WORKDIR:-${trained_root}/sudoku_lora_7256999}"
checkpoint_step="${SUDOKU_EVAL_STEP:-2000}"
wandb_project="diffgemma-sudoku"
wandb_mode="online"
lora_rank=8
train_steps=2000

if [[ ! "$checkpoint_step" =~ ^[0-9]+$ ]]; then
    echo "SUDOKU_EVAL_STEP must be a nonnegative checkpoint step." >&2
    exit 1
fi
checkpoint_dir="${workdir}/checkpoints/ckpt_${checkpoint_step}"
if [[ ! -f "${checkpoint_dir}/commit_success.txt" ]]; then
    echo "Completed training checkpoint not found: ${checkpoint_dir}" >&2
    exit 1
fi
for file in "${data_root}/sudoku_train.bagz" \
    "${data_root}/sudoku_eval.bagz" \
    "${persist_root}/gemma_cache/tokenizer/tokenizer_gemma4.model"; do
    if [[ ! -s "$file" ]]; then
        echo "Required data/tokenizer file not found or empty: ${file}" >&2
        exit 1
    fi
done

# Global batch size, sharded over the allocated GPUs, so it must be divisible by
# their count (a full GH200 node has 4: use --gpus=4 and e.g. batch_size=4).
num_gpus="${SLURM_GPUS_ON_NODE:-${SLURM_GPUS:-1}}"
batch_size=2
if (( batch_size % num_gpus != 0 )); then
    echo "batch_size=${batch_size} is not divisible by num_gpus=${num_gpus}" >&2
    exit 1
fi
echo "GPUs: ${num_gpus}, global batch size: ${batch_size}, workdir: ${workdir}, checkpoint step: ${checkpoint_step}"


# ---------------------------------------------------------------------------
# main script
# ---------------------------------------------------------------------------

cd "${repo_dir}"

# The launcher forwards XLA_FLAGS, NCCL_* and CUDA_VISIBLE_DEVICES into the container.
env XLA_FLAGS="--xla_disable_hlo_passes=constant_folding" \
    XLA_PYTHON_CLIENT_PREALLOCATE="false" \
    APPTAINERENV_TF_FORCE_GPU_ALLOW_GROWTH="true" \
    NCCL_ALGO="Ring" \
    NCCL_PROTO="LL128" \
    NCCL_NVLS_ENABLE="0" \
    NCCL_CUMEM_ENABLE="0" \
    srun --nodes=1 --ntasks=1 --gpus="${num_gpus}" \
    scripts/isambard_cuda13.sh python -m gemma.diffusion.hackable_diffusion_adapter.eval_main \
    --cfg=gemma/diffusion/hackable_diffusion_adapter/configs/sft_sudoku.py \
    --task=sudoku \
    --step="${checkpoint_step}" \
    --eval_names=sample_ar_steps64 \
    --cfg.aux.eval_num_batches=2 \
    --cfg.aux.num_canvases=2 \
    --cfg.workdir="${workdir}" \
    --cfg.aux.checkpoint_path="${pretrained_root}" \
    --cfg.train_ds.bagz_path="${data_root}/sudoku_train.bagz" \
    --cfg.eval_ds.bagz_path="${data_root}/sudoku_eval.bagz" \
    --cfg.train_ds.batch_size="${batch_size}" \
    --cfg.eval_ds.batch_size="${batch_size}" \
    --cfg.writer.wbproj="${wandb_project}" \
    --cfg.writer.mode="${wandb_mode}" \
    --cfg.aux.lora_rank="${lora_rank}" \
    --cfg.num_train_steps="${train_steps}"
