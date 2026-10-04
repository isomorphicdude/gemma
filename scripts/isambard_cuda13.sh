#!/usr/bin/env bash
# Run the checkout's ARM64 virtual environment with CUDA forward compatibility.
set -euo pipefail

repo_root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd -P)
image=${GEMMA_CUDA13_IMAGE:-"$repo_root/.apptainer/cuda-13.0.0-arm64.sif"}
# NVIDIA cuda:13.0.0-base-ubuntu24.04, pinned to its Linux ARM64 manifest.
# Its R580 compat driver can initialize on Isambard's R565 kernel driver.
image_uri='docker://nvcr.io/nvidia/cuda@sha256:c33da1d7a948e960842549e344af32eed7d8be20d1922ff99b52c119b08c72a7'

usage() {
  cat <<'EOF'
Usage: scripts/isambard_cuda13.sh [--pull | --check | COMMAND [ARG...]]

  --pull   Download the pinned image on a login node (two compression threads).
  --check  Check the driver libraries, Gemma import, and a JAX GPU computation.
  COMMAND  Run a command with the checkout's .venv active (default: bash).

Run GPU commands inside a Slurm allocation, for example:
  srun --nodes=1 --gpus=1 --ntasks=1 --time=00:05:00 scripts/isambard_cuda13.sh --check

Set GEMMA_CUDA13_IMAGE to store/use the SIF at a different absolute path.
EOF
}

case "${1:-}" in
  --help|-h) usage; exit 0 ;;
esac

if [[ $(uname -m) != aarch64 ]]; then
  echo 'This launcher requires an ARM64/aarch64 host.' >&2
  exit 2
fi
if ! command -v apptainer >/dev/null; then
  echo 'Apptainer is required; it is installed on Isambard-AI Phase 2.' >&2
  exit 2
fi

if [[ ${1:-} == --pull ]]; then
  if [[ -s "$image" ]]; then
    printf 'Image already exists: %s\n' "$image"
    exit 0
  fi
  mkdir -p -- "$(dirname -- "$image")"
  # building image using apptainer
  exec apptainer build --mksquashfs-args '-processors 2' "$image" "$image_uri"
fi

if [[ ! -s "$image" ]]; then
  echo 'Download the container first: scripts/isambard_cuda13.sh --pull' >&2
  exit 2
fi
if [[ ! -x "$repo_root/.venv/bin/python" ]]; then
  echo 'Install the ARM64 .venv described in FORK_SETUP.md first.' >&2
  exit 2
fi
if [[ -z ${SLURM_JOB_ID:-} ]]; then
  echo 'Run GPU commands through srun or inside an existing Slurm allocation.' >&2
  exit 2
fi

case "${1:-}" in
  --check) set -- python -u "$repo_root/scripts/check_cuda13.py" ;;
  --) shift ;;
esac
if (( $# == 0 )); then
  set -- bash --noprofile --norc
fi

options=(--nv --cleanenv --no-eval --pwd "$repo_root"
  --bind "$repo_root:$repo_root" --bind "$HOME:$HOME")
if [[ -d /projects ]]; then
  options+=(--bind /projects:/projects)
fi
# --cleanenv excludes host CUDA modules, PYTHONPATH, and activated environments.
# Preserve Slurm's GPU selection and the documented training tuning variables.
for name in CUDA_VISIBLE_DEVICES XLA_FLAGS JAX_PLATFORMS \
  JAX_COMPILATION_CACHE_DIR XLA_PYTHON_CLIENT_PREALLOCATE \
  XLA_PYTHON_CLIENT_MEM_FRACTION XLA_PYTHON_CLIENT_ALLOCATOR \
  NCCL_ALGO NCCL_PROTO NCCL_NVLS_ENABLE NCCL_CUMEM_ENABLE; do
  if [[ -v $name ]]; then
    export "APPTAINERENV_${name}=${!name}"
  fi
done

exec apptainer exec "${options[@]}" "$image" /bin/bash --noprofile --norc -c '
  compat=/usr/local/cuda/compat
  if [[ ! -r "$compat/libcuda.so.1" ]]; then
    compat=/usr/local/cuda-13.0/compat
  fi
  if [[ ! -r "$compat/libcuda.so.1" ]]; then
    echo "The container is missing CUDA forward compatibility libraries." >&2
    exit 2
  fi
  # --nv injects the older host driver into /.singularity.d/libs. Prefer compat.
  # CUDA runtime/math libraries come from the pinned Python wheels. Exclude the
  # image runtime lib64 directory so it cannot override those newer libraries.
  export LD_LIBRARY_PATH="$compat:/.singularity.d/libs"
  export VIRTUAL_ENV="$1"
  export PATH="$VIRTUAL_ENV/bin:$PATH"
  export PYTHONNOUSERSITE=1
  unset PYTHONHOME PYTHONPATH
  shift
  exec "$@"
' gemma-cuda13 "$repo_root/.venv" "$@"
