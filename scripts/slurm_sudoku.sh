#!/bin/bash

#SBATCH --job-name=sudoku-sft
#SBATCH --output=/projects/u6tp/tw1320/submit/out/sudoku_%A_%a.out
#SBATCH --error=/projects/u6tp/tw1320/submit/out/sudoku_%A_%a.err
# Single node only: one JAX process shards over all of the node's allocated GPUs.
# Without --nodes=1, --gpus=2 may be split across two nodes.
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

# check all dirs exist
for dir in "$data_root" "$pretrained_root" "$trained_root"; do
    mkdir -p "$dir"
done


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

workdir="${trained_root}/sudoku_lora_${SLURM_JOB_ID:-local}"
wandb_project="diffgemma-sudoku"
wandb_mode="online"
lora_rank=8
train_steps=2000

# Global batch size, sharded over the allocated GPUs, so it must be divisible by
# their count (a full GH200 node has 4: use --gpus=4 and e.g. batch_size=4).
num_gpus="${SLURM_GPUS_ON_NODE:-${SLURM_GPUS:-1}}"
batch_size=2
if (( batch_size % num_gpus != 0 )); then
    echo "batch_size=${batch_size} is not divisible by num_gpus=${num_gpus}" >&2
    exit 1
fi
echo "GPUs: ${num_gpus}, global batch size: ${batch_size}, workdir: ${workdir}"


# ---------------------------------------------------------------------------
# main script
# ---------------------------------------------------------------------------

cd "${repo_dir}"

# The launcher forwards XLA_FLAGS, NCCL_* and CUDA_VISIBLE_DEVICES into the container.
env XLA_FLAGS="--xla_disable_hlo_passes=constant_folding" \
    NCCL_ALGO="Ring" \
    NCCL_PROTO="LL128" \
    NCCL_NVLS_ENABLE="0" \
    NCCL_CUMEM_ENABLE="0" \
    srun --nodes=1 --ntasks=1 --gpus="${num_gpus}" \
    scripts/isambard_cuda13.sh python -m kauldron.main \
  --cfg=gemma/diffusion/hackable_diffusion_adapter/configs/sft_sudoku.py \
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
