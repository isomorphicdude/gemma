# Hackable Diffusion Adapter — Agent Playbook

This document is designed to guide AI coding agents through the structure,
setup, testing, and training workflows of the **Hackable Diffusion (HD) Text
Diffusion Supervised Fine-Tuning (SFT)** adapter library.

> **Fork note:** this is the `isomorphicdude/gemma` fork (work branch:
> `sudoku-sft`). The installation steps below already follow the fork
> procedure (editable install with `constraints.txt`); see
> [FORK_SETUP.md](../../../FORK_SETUP.md) at the repo root for the rest of
> the fork setup (git workflow, arm64 / GH200 details, offline checkpoint
> and tokenizer provisioning, data copying, known limitations). Push to
> `origin` (the fork) only, never to `upstream`.

---

## 📋 Codebase Structure

The project is structured as a standard Python/JAX package.

*   **`configs/`**: Kauldron config files defining task hyperparameters, datasets, losses, optimizers, and evaluators.
    *   [`sft_sudoku.py`](configs/sft_sudoku.py): LoRA-based SFT training for the Sudoku puzzle solving task.
    *   [`sft_sudoku_full.py`](configs/sft_sudoku_full.py): Full weight SFT training for the Sudoku puzzle solving task (no LoRA).
    *   [`sft_pubmedqa.py`](configs/sft_pubmedqa.py): LoRA-based SFT training for the PubMedQA long-answer task.
*   **`data/`**: Dataset loading, custom pipelines, and preprocessing transforms.
    *   [`data.py`](data/data.py): Common transforms (e.g. `CanvasChunker` for localized diffusion).
*   **`hd/`**: Core modeling, network layers, and state handling.
    *   [`sft_model.py`](hd/sft_model.py): Core `SFTDiffusion` class managing the hybrid AR prefill and localized diffusion denoising steps.
    *   [`lora.py`](hd/lora.py): PEFT LoRA wrappers.
    *   [`mask_helpers.py`](hd/mask_helpers.py): Right-pad causal/block masks and cursor tracking.
*   **`eval/`**: Custom evaluation metrics designed to avoid TPU/GPU OOM issues.
    *   [`sudoku_eval.py`](eval/sudoku_eval.py): Unified host-side `SudokuAllMetrics` evaluation.

---

## 🛠️ Setup and Installation

These instructions target the **arm64 / GH200 remote cluster**. **Do not**
`pip install gemma` from PyPI and do not use a plain `pip install .`: both
copy a frozen snapshot into `site-packages`, so edits in the checkout are
silently ignored. This fork is installed in **editable mode** with pinned
constraints, so code edits are live without reinstalling (moving between
machines is just commit / push / pull). See
[FORK_SETUP.md](../../../FORK_SETUP.md) for the full fork setup (git
workflow, offline checkpoint/tokenizer provisioning, data copying, known
limitations).

### 1. Prerequisites

*   Python 3.12 and [uv](https://docs.astral.sh/uv/) — needs a version with
    `--excludes` (tested with 0.11.28). Plain pip has no equivalent of
    `--excludes`; use uv.
*   A driver compatible with the CUDA 13 wheels, either natively (580+) or
    through the validated Apptainer setup below. Isambard Phase 2 currently
    has driver 565.57.01; loading its CUDA 11.8/12.6 modules is insufficient.
*   Apptainer for the Isambard launcher (tested with 1.4.1).
*   glibc 2.28 or newer (`ldd --version`).
*   Internet access during provisioning. The current ARM64 environment has
    `bagz` 0.3.8 installed. If an installation needs to build `bagz` from
    source, provide a C++20-capable compiler and access to Abseil/zstd
    downloads; do not assume every ARM64 installation requires this build.

### 2. Clone the fork and create the environment

The virtual environment lives inside the clone at `.venv/` (git-ignored), so
one directory holds code, environment and data:

```bash
git clone -b sudoku-sft git@github.com:isomorphicdude/gemma.git
cd gemma
uv venv --python 3.12 --prompt diffgemma
source .venv/bin/activate
```

### 3. Editable install with constraints

```bash
uv pip install -e . "jax[cuda13]" -c constraints.txt --excludes excludes-aarch64.txt
```

`--excludes excludes-aarch64.txt` drops `tensorflow-cpu`, which has no arm64
build in any version (the full `tensorflow` wheel exists for arm64 and is
pulled in anyway by `tensorflow-text`). On x86-64 machines, omit
`--excludes`.

If another environment is active in the shell, uv installs into that one
(`$VIRTUAL_ENV` wins over `./.venv`); either `deactivate` first or add
`--python .venv/bin/python` to the install command.

### 4. Isambard Phase 2: provision on login, run through Apptainer

Use [the project launcher](../../../scripts/isambard_cuda13.sh) for GPU
commands on this cluster. Run the following from the **repository root**,
the directory containing `pyproject.toml`, `.venv/`, `scripts/`, and `gemma/`:

```bash
# Download/build once on a login node; subsequent calls reuse the cached SIF.
scripts/isambard_cuda13.sh --pull

# Allocate two GPUs to check both CUDA execution and NCCL communication.
XLA_FLAGS="--xla_disable_hlo_passes=constant_folding" \
NCCL_ALGO=Ring NCCL_PROTO=LL128 NCCL_NVLS_ENABLE=0 NCCL_CUMEM_ENABLE=0 \
srun --nodes=1 --gpus=2 --ntasks=1 --time=00:05:00 \
    scripts/isambard_cuda13.sh --check
```

Compute-node downloads are slow. Prepare the image on the login node before
starting GPU jobs; `--pull` limits image compression to two CPU threads.
The default cache is `.apptainer/cuda-13.0.0-arm64.sif` (git-ignored).
To use project storage, set `GEMMA_CUDA13_IMAGE` to the same absolute SIF path
for provisioning and execution. Stage checkpoints and tokenizers before
training as described in [FORK_SETUP.md](../../../FORK_SETUP.md).

The launcher pins NVIDIA `cuda:13.0.0-base-ubuntu24.04` by its ARM64 manifest
digest. Its R580 compatibility `libcuda` works with the installed JAX 0.11.2
and CUDA 13.4.92 wheels. The CUDA 13.4.1 image's R615 compatibility driver
failed `cuInit` with `CUDA_ERROR_SYSTEM_DRIVER_MISMATCH` on this cluster;
do not replace the image with a newer tag without a GPU workload check.

Apptainer runs with `--nv --cleanenv --no-eval`. Inside the container,
`LD_LIBRARY_PATH=/usr/local/cuda/compat:/.singularity.d/libs` selects the
compatibility driver before the injected host driver. CUDA runtime/math
libraries come from the pinned `.venv` wheels, so the image's CUDA runtime
directory is excluded. Keep this library path local to the launcher;
do not add it to shell profiles or prepend system CUDA module paths.
The launcher uses the editable `.venv`, binds the checkout/home/`/projects`,
and preserves Slurm's GPU selection and the XLA/NCCL settings shown above.
Respect the allocation's `CUDA_VISIBLE_DEVICES`; do not replace it with a
hard-coded GPU index. Use `APPTAINERENV_<NAME>` for additional application
variables that the launcher does not explicitly forward.

Verified on 2026-10-04: `libcuda.so.580.65.06`, the wheel-provided CUDA 13.4
runtime, a JAX GPU matrix product, a two-GPU NCCL/JAX all-reduce, and all nine
Sudoku config tests. The GPU check forces the CUDA backend and checks
`cuInit`, imported Gemma location, and computed results; a driver API version
or device listing alone is insufficient. Full training and multi-node jobs
remain untested. See [FORK_SETUP.md](../../../FORK_SETUP.md#isambard-ai-phase-2-apptainer-launcher)
for the compatibility support caveat and detailed validation record.

### 5. Verify the editable import and Sudoku config

```bash
# From the repository root, must print a path inside the editable clone.
.venv/bin/python -c "import importlib.util as u; print(u.find_spec('gemma').origin)"
# Isambard: run all nine Sudoku config tests through the launcher, on CPU.
JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES="" \
srun --nodes=1 --gpus=1 --ntasks=1 --time=00:05:00 \
    scripts/isambard_cuda13.sh python -m \
    gemma.diffusion.hackable_diffusion_adapter.configs.sft_sudoku_test
```

> **CUDA package consistency:** keep the existing CUDA 13 environment and
> `constraints.txt` pins. Do not mix in CUDA 12 plugins or libraries such as
> `jax-cuda12-plugin` or `nvidia-nccl-cu12`, or downgrade JAX/CUDA merely to
> match Isambard's native driver. Use the validated compatibility launcher
> and repeat its GPU check when changing the image or CUDA wheels.

---

## 💾 Dataset Preparation

Before launching SFT training, datasets must be preprocessed.
Note that running these scripts requires first cloning the source repository.

### PubMedQA Dataset
```bash
cd gemma/diffusion/hackable_diffusion_adapter/data/pubmedqa
bash prepare_pubmedqa_dataset.sh
cd -
```

### Sudoku Dataset
Requires Kaggle API access token. Ask the user to generate an access token and
then run the following command.

```bash
mkdir -p ~/.kaggle && echo YOUR_KAGGLE_TOKEN > ~/.kaggle/access_token && chmod 600 ~/.kaggle/access_token
```

Then the datapipeline can be run with the following command

```bash
cd gemma/diffusion/hackable_diffusion_adapter/data/sudoku
bash prepare_sudoku_dataset.sh
cd -
```

---

## 🧪 Running Unit Tests

The Sudoku config checks above use the existing environment and do not
require pytest. For pytest suites, install the development dependencies if
needed, retaining the fork's constraints and ARM64 exclusion:

```bash
uv pip install --python .venv/bin/python -e ".[dev]" "jax[cuda13]" \
    -c constraints.txt --excludes excludes-aarch64.txt
```

To verify the JAX layers, data pipelines, and sampling routines, run:

```bash
# Inside a Slurm allocation on Isambard, from the repository root.
scripts/isambard_cuda13.sh python -m pytest gemma/diffusion/hackable_diffusion_adapter/
```

Or run individual tests:
```bash
scripts/isambard_cuda13.sh python -m pytest gemma/diffusion/hackable_diffusion_adapter/hd/lora_test.py
```

---

## 🚀 Launching SFT Training

Always use the standard Kauldron CLI command. Use the following env variables
to prevent JIT compilation OOMs and NCCL communication hangs:

Run from the repository root. On Isambard, use the cached image and the
launcher through Slurm, as below. These examples allocate two GPUs on one
node; global batch sizes must be divisible by the allocated GPU count.
Ensure the prepared datasets, checkpoint, and tokenizer are available
before starting a training allocation.

#### PubMedQA

```bash
env XLA_FLAGS="--xla_disable_hlo_passes=constant_folding" \
    NCCL_ALGO="Ring" \
    NCCL_PROTO="LL128" \
    NCCL_NVLS_ENABLE="0" \
    NCCL_CUMEM_ENABLE="0" \
    srun --nodes=1 --gpus=2 --ntasks=1 \
    scripts/isambard_cuda13.sh python -m kauldron.main \
  --cfg=gemma/diffusion/hackable_diffusion_adapter/configs/sft_pubmedqa.py \
  --cfg.workdir=$(pwd)/xp_dir
```

#### Sudoku (with LoRA)

```bash
env XLA_FLAGS="--xla_disable_hlo_passes=constant_folding" \
    NCCL_ALGO="Ring" \
    NCCL_PROTO="LL128" \
    NCCL_NVLS_ENABLE="0" \
    NCCL_CUMEM_ENABLE="0" \
    srun --nodes=1 --gpus=2 --ntasks=1 \
    scripts/isambard_cuda13.sh python -m kauldron.main \
  --cfg=gemma/diffusion/hackable_diffusion_adapter/configs/sft_sudoku.py \
  --cfg.workdir=$(pwd)/xp_dir
```

#### Sudoku (full weight updates)

```bash
env XLA_FLAGS="--xla_disable_hlo_passes=constant_folding" \
    NCCL_ALGO="Ring" \
    NCCL_PROTO="LL128" \
    NCCL_NVLS_ENABLE="0" \
    NCCL_CUMEM_ENABLE="0" \
    srun --nodes=1 --gpus=2 --ntasks=1 \
    scripts/isambard_cuda13.sh python -m kauldron.main \
  --cfg=gemma/diffusion/hackable_diffusion_adapter/configs/sft_sudoku_full.py \
  --cfg.workdir=$(pwd)/xp_dir
```

---

## 📊 Offline Evaluation

Evaluation is run **offline** — it is a separate step from training. The
`eval_main` binary loads a saved checkpoint, runs AR diffusion sampling
on the eval dataset, and reports task-specific metrics.

### Running an Eval Job

From the repository root on Isambard, using the same cached image:

```bash
env XLA_FLAGS="--xla_disable_hlo_passes=constant_folding" \
    XLA_PYTHON_CLIENT_PREALLOCATE="false" \
    APPTAINERENV_TF_FORCE_GPU_ALLOW_GROWTH="true" \
    NCCL_ALGO="Ring" \
    NCCL_PROTO="LL128" \
    NCCL_NVLS_ENABLE="0" \
    NCCL_CUMEM_ENABLE="0" \
    srun --nodes=1 --gpus=2 --ntasks=1 \
    scripts/isambard_cuda13.sh python -m gemma.diffusion.hackable_diffusion_adapter.eval_main \
    --cfg=gemma/diffusion/hackable_diffusion_adapter/configs/sft_sudoku.py \
    --task=sudoku \
    --step=1000 \
    --eval_names=sample_ar_steps64 \
    --cfg.workdir=$(pwd)/xp_dir_sudoku_lora \
    --cfg.eval_ds.batch_size=2 \
    --cfg.aux.eval_num_batches=2 \
    --cfg.aux.num_canvases=2
```

### Key Flags

| Flag | Description |
|---|---|
| `--cfg` | Path to the training config file (same one used for training). |
| `--task` | Task to evaluate: `sudoku` or `pubmedqa`. Determines which metrics are reported. |
| `--step` | Checkpoint step to evaluate. If omitted, the latest checkpoint is used. |
| `--eval_names` | Comma-separated list of evaluators to run (e.g. `sample_ar_steps64`). If omitted, all evaluators are run. |
| `--cfg.workdir` | Working directory containing the training checkpoints. |
| `--cfg.eval_ds.batch_size` | Eval batch size (reduce if running out of memory). |
| `--cfg.aux.eval_num_batches` | Number of eval batches to process. Set to a small value for quick sanity checks, or omit to run over the full eval set. |
| `--cfg.aux.num_canvases` | Number of AR canvases to generate per example. |

### Available Evaluators

Evaluators are generated automatically by
[`ar_eval.make_ar_evals`](eval/ar_eval.py).
The naming convention is:

*   `sample_ar_steps{N}` — AR sampling with `N` denoising steps
    (default values: 32, 64, 96).
*   `sample_ar_steps{N}_early_stopping` — Same as above but with
    entropy-based early stopping.

### Checking Eval Results

Eval metrics are written to TensorBoard event files in the working directory.
Launch TensorBoard to view them:

```bash
tensorboard --logdir=$(pwd)/xp_dir_sudoku_lora
```

Metrics for each evaluator appear under the corresponding eval name
(e.g. `sample_ar_steps64`). Key metrics by task:

*   **Sudoku**: overall accuracy, cell accuracy, difficulty-stratified results
    (easy/medium/hard), exact mask accuracy.
*   **PubMedQA**: short-answer accuracy, BLEU score.

---

## 🚫 Guardrails & Best Practices for Code Modifications

*   **No Raw `jnp.roll` or pad index assumptions**: When shifting sequences or
    tracking attention masks, always use utilities from `mask_helpers.py`
    (e.g. `set_cache_end_index`) to ensure compatibility with right-pad
    conventions.
*   **Keep configs flat**: Do not import helper config modules inside configs. Keep them self-contained and editable as single drop-in files.
*   **Evaluate on host side for complex metrics**: Custom evaluation metrics
    MUST inherit from `BaseTextMetric` or `BaseSimpleTextMetric` and execute
    calculations inside `io_callback` on CPU to prevent device-side OOM.
