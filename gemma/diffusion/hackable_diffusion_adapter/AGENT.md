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
    *   [`sft_nurikabe.py`](configs/sft_nurikabe.py): LoRA-based SFT training for 10x10 Nurikabe (reuses the Sudoku Bagz pipeline with its own prompt).
*   **`data/`**: Dataset loading, custom pipelines, and preprocessing transforms.
    *   [`data.py`](data/data.py): Common transforms (e.g. `CanvasChunker` for localized diffusion).
*   **`hd/`**: Core modeling, network layers, and state handling.
    *   [`sft_model.py`](hd/sft_model.py): Core `SFTDiffusion` class managing the hybrid AR prefill and localized diffusion denoising steps.
    *   [`lora.py`](hd/lora.py): PEFT LoRA wrappers.
    *   [`mask_helpers.py`](hd/mask_helpers.py): Right-pad causal/block masks and cursor tracking.
*   **`eval/`**: Custom evaluation metrics designed to avoid TPU/GPU OOM issues.
    *   [`sudoku_eval.py`](eval/sudoku_eval.py): Unified host-side `SudokuAllMetrics` evaluation.
    *   [`nurikabe_eval.py`](eval/nurikabe_eval.py): Host-side `NurikabeAllMetrics` and `NurikabeBoardImages`, scored with [`nurikabe_verify.py`](eval/nurikabe_verify.py) and drawn with [`nurikabe_viz.py`](eval/nurikabe_viz.py).

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
    through the validated Apptainer setup below. Isambard Phase 2 compute
    nodes run driver 580.173.02 since 2026-10-07 (565.57.01 before); loading
    its CUDA 11.8/12.6 modules is insufficient.
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
training; see [step 6](#6-isambard-staged-checkpoint-and-tokenizer).

The launcher pins NVIDIA `cuda:13.0.0-base-ubuntu24.04` by its ARM64 manifest
digest. Its R580 compatibility `libcuda` (580.65.06) and the host's 580.173.02
driver both work with the installed JAX 0.11.2 and CUDA 13.4.92 wheels. The
CUDA 13.4.1 image's R615 compatibility driver
failed `cuInit` with `CUDA_ERROR_SYSTEM_DRIVER_MISMATCH` on this cluster;
do not replace the image with a newer tag without a GPU workload check.

Apptainer runs with `--nv --cleanenv --no-eval`. Inside the container, the
launcher puts `/usr/local/cuda/compat` before the injected host driver
(`/.singularity.d/libs`) only when the compatibility driver is newer than the
host kernel driver; otherwise `LD_LIBRARY_PATH=/.singularity.d/libs`. A
compatibility driver older than the kernel driver fails `cuInit` with
`CUDA_ERROR_SYSTEM_DRIVER_MISMATCH` and JAX falls back to CPU, which broke
the Sudoku jobs after the 2026-10-07 driver update. CUDA runtime/math
libraries come from the pinned `.venv` wheels, so the image's CUDA runtime
directory is excluded. Keep this library path local to the launcher;
do not add it to shell profiles or prepend system CUDA module paths.
The launcher uses the editable `.venv`, binds the checkout/home/`/projects`,
and preserves Slurm's GPU selection and the XLA/NCCL settings shown above.
Respect the allocation's `CUDA_VISIBLE_DEVICES`; do not replace it with a
hard-coded GPU index. Use `APPTAINERENV_<NAME>` for additional application
variables that the launcher does not explicitly forward.

The GPU check forces the CUDA backend and checks `cuInit`, imported Gemma
location, and computed results; a driver API version or device listing alone
is insufficient. What has passed on this cluster is in the
[verification record](#-verification-record); full training and multi-node
jobs remain untested. See
[FORK_SETUP.md](../../../FORK_SETUP.md#isambard-ai-phase-2-apptainer-launcher)
for the compatibility support caveat.

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
> `jax-cuda12-plugin` or `nvidia-nccl-cu12`. Keep `etils` on the commit pinned
> in `pyproject.toml`: the PyPI 1.14.0 release fails with jax 0.11
> (`module 'jax._src' has no attribute 'prng'`). Use the launcher and repeat
> its GPU check when changing the image or CUDA wheels, and after a cluster
> driver update.

### 6. Isambard: staged checkpoint and tokenizer

Both are staged in project storage, visible from the compute nodes (generic
offline provisioning steps are in
[FORK_SETUP.md](../../../FORK_SETUP.md#base-checkpoint-and-tokenizer)):

*   Checkpoint: downloaded on the login node on 2026-10-05 with
    `gcloud storage rsync -r gs://gemma-data/checkpoints/diffusiongemma-26B-A4B-it /projects/u6tp/tw1320/diffgemma/gemma_weights/`
    (the contents sit directly in `gemma_weights/`). Pass
    `--cfg.aux.checkpoint_path=/projects/u6tp/tw1320/diffgemma/gemma_weights`.
*   Tokenizer: stage it next to the weights so the run does not depend on
    home-directory visibility from the compute nodes:

    ```bash
    mkdir -p /projects/u6tp/tw1320/diffgemma/gemma_cache/tokenizer
    curl -o /projects/u6tp/tw1320/diffgemma/gemma_cache/tokenizer/tokenizer_gemma4.model \
        https://storage.googleapis.com/gemma-data/tokenizers/tokenizer_gemma4.model
    # The launcher's --cleanenv drops GEMMA_CACHE_DIR; forward it explicitly:
    export APPTAINERENV_GEMMA_CACHE_DIR=/projects/u6tp/tw1320/diffgemma/gemma_cache
    ```

[`scripts/slurm_sudoku.sh`](../../../scripts/slurm_sudoku.sh) already passes
both.

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

### Nurikabe Dataset
Public on the Hugging Face Hub; needs the `hf` CLI but no token. From the
repository root, this downloads the JSONL splits next to the converter and
writes `nurikabe_{train,eval,test}.bagz` there (see
[data/README.md](data/README.md#converting-nurikabe)):

```bash
PATH="$(pwd)/.venv/bin:$PATH" \
    bash gemma/diffusion/hackable_diffusion_adapter/data/nurikabe/prepare_nurikabe_dataset.sh
```

On Isambard, [`scripts/slurm_nurikabe.sh`](../../../scripts/slurm_nurikabe.sh)
reads the Bagz files from `/projects/u6tp/tw1320/diffgemma/nurikabe_data/`;
copy them there after converting.

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

#### Nurikabe (with LoRA)

[`scripts/slurm_nurikabe.sh`](../../../scripts/slurm_nurikabe.sh) passes the
staged checkpoint, data and tokenizer paths. Extra arguments are forwarded to
`kauldron.main`, so a short run without checkpoints (37 GiB each) is:

```bash
NURIKABE_WORKDIR=/projects/u6tp/tw1320/diffgemma/trained/nurikabe_smoke \
sbatch --time=00:40:00 scripts/slurm_nurikabe.sh \
    --cfg.num_train_steps=150 --cfg.log_metrics_every=10 \
    '--cfg.checkpointer=py::kauldron.checkpoints:NoopCheckpointer()' \
    '--cfg.evals=py::{}' --cfg.writer.mode=disabled
```

`num_train_steps` must exceed the 100 warmup steps, or the cosine schedule
fails with `decay_steps=0`. Without arguments it trains 2000 steps.

For a full run followed by evaluation, submit both jobs at once:

```bash
bash scripts/submit_nurikabe.sh
```

It creates `trained/nurikabe_lora_<UTC time>_<suffix>/`, trains into it, and
queues [`scripts/slurm_nurikabe-eval.sh`](../../../scripts/slurm_nurikabe-eval.sh)
with `afterok`. The eval job restores the newest committed checkpoint
(`NURIKABE_EVAL_STEP=latest`, or a step number) and runs
`eval_main --task=nurikabe` with `sample_ar_steps64` on the 256-example eval
slice (42 min on two GPUs; `NURIKABE_EVAL_NUM_BATCHES` shortens it). Both jobs log to the same wandb run in `diffgemma-nurikabe`,
because the run ID is derived from the workdir. The eval script can also be
submitted alone with `NURIKABE_WORKDIR=<run dir>`.

The run size defaults to 2000 steps at global batch 2 on two GPUs. Set
`NURIKABE_TRAIN_STEPS`, `NURIKABE_BATCH_SIZE` (global) and
`NURIKABE_TRAIN_GPUS` to change it; both jobs read the step count, which is
part of the wandb run name. For example, 30K examples on a full node, still
one example per GPU:

```bash
NURIKABE_TRAIN_STEPS=7500 NURIKABE_BATCH_SIZE=4 NURIKABE_TRAIN_GPUS=4 \
  bash scripts/submit_nurikabe.sh
```

Each checkpoint is about 37 GB, and the checkpointer keeps the last five.

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
| `--task` | Task to evaluate: `sudoku`, `pubmedqa` or `nurikabe`. Determines which metrics are reported. |
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
*   **Nurikabe**: `nurikabe_accuracy` (shading accepted by
    `nurikabe_verify.verify`), exact match including island sizes, shading
    accuracy on unknown cells, clue preservation, well-formed output rate;
    `nurikabe_boards` images.

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

---

## ✅ Verification Record

What has been checked, where, and when. Append new results here rather than
in FORK_SETUP.md.

### Fork baseline

*   `main` last synced with upstream at `c490bec` (2026-10-02).


### arm64 static checks

*   2026-10-02: PyPI metadata for every pinned package, a uv resolution
    targeting `aarch64-manylinux_2_28` / Python 3.12, and the `bagz` build
    files. Result: installs once `tensorflow-cpu` is excluded; `bagz`
    compiles from source; all other 264 packages have arm64 wheels.

### Isambard-AI Phase 2

*   Driver history: GH200 compute nodes ran 565.57.01 (native CUDA through
    12.7) until 2026-10-07. They were rebooted 2026-10-07 to 09; node
    `nid010046` reported 580.173.02 (native CUDA 13.0) on 2026-10-11.
*   The CUDA 13.4.1 image was tested: its R615 compatibility driver failed
    `cuInit` on the R565 kernel driver with
    `CUDA_ERROR_SYSTEM_DRIVER_MISMATCH`.
*   2026-10-04, node `nid010327`: the launcher selected
    `libcuda.so.580.65.06` (driver API 13.0) and the wheel-provided CUDA 13.4
    runtime (13040). JAX 0.11.2 listed two `CudaDevice` instances and passed a
    compiled matrix product and an all-reduce across the two GPUs. An
    inherited host CUDA 12.6 library path was deliberately set for this
    check; the container used only its compatibility and injected driver
    library directories. All nine Sudoku config and data-pipeline structure
    tests passed through the launcher with the CPU backend; `bagz` 0.3.8 is
    installed in the ARM64 `.venv`. These checks did not load model weights
    or train the model.
*   2026-10-11, node `nid010046`, after the driver update: the previous
    launcher always preferred the compatibility driver, so JAX silently fell
    back to CPU. The compatibility `libcuda.so.580.65.06` failed `cuInit` with
    `CUDA_ERROR_SYSTEM_DRIVER_MISMATCH`; the same image with only the host
    `libcuda.so.580.173.02` listed a `CudaDevice` and passed the compiled
    matrix product. The updated launcher's `--check` then passed.
*   2026-10-11: the first Slurm training attempts (jobs 7254375 and 7254801)
    failed before the first step, from the CUDA driver mismatch above and the
    `etils` `jax._src.prng` error. Both are fixed (updated launcher, `etils`
    pin); no training run has completed since.
*   2026-10-16: base-model sampling on one GH200 through the launcher
    (jobs 7486185 and 7486297): the checkpoint loaded from the staged copy,
    and a 512-token canvas was denoised with 48 time points in 73.5 s. Every
    run logged a recovered 12 GiB allocator retry. The trajectory video
    notebook and the `diffusiongemma` Jupyter kernel were checked as well;
    see the run record in [TRAJECTORY_VIDEO.md](TRAJECTORY_VIDEO.md).
*   Untested: full training runs and multi-node jobs.
*   2026-10-21, node `nid010130`, current GPU allocation 7605020:
    LLaDA-8B-Instruct masked sampling passed in a separate PyTorch
    environment (129 states, 256-token canvas, 3.57 s sampling, 15.21 GiB
    peak allocation). Both masked and uniform trajectory rendering passed;
    the masked sampler also passed through the existing CUDA launcher;
    the nine Sudoku config tests passed through the launcher on CPU. See
    [TRAJECTORY_VIDEO.md](TRAJECTORY_VIDEO.md#masked-trajectory-2026-10-21)
    for settings, artifacts, and the verification record.
*   2026-10-21, same allocation: MD4 linear ancestral reveal sampling
    passed on the GH200 with both categorical and greedy tokens (3.77 s
    and 3.64 s). Distribution checks confirmed confidence-independent
    position selection, linear reveal rates, and fixed revealed tokens;
    both presentation exports passed. Settings and comparison curves are
    in [TRAJECTORY_VIDEO.md](TRAJECTORY_VIDEO.md#md4-linear-sampling).
*   2026-10-27, Nurikabe data: `isometricneko/nurikabe10x10` (revision
    `8ccd4b2`) converted to 45,023 / 2,487 / 2,490 train/eval/test records
    and staged in `/projects/u6tp/tw1320/diffgemma/nurikabe_data/`. Every
    record round-trips from JSONL, every solution passes
    `nurikabe_verify.verify`, and the largest prompt and response are 215
    and 154 tokens (budgets 256 and 256). The 8 Nurikabe config tests and
    19 eval tests passed on the login node's CPU.
*   2026-10-27, job 7773424 on `nid010110`, two GPUs: a 150-step Nurikabe
    LoRA smoke run through `scripts/slurm_nurikabe.sh` (no checkpoints,
    wandb disabled) completed in 17 min 40 s. Total loss went from 4.26 to
    0.47 (encoder 2.89 → 0.25, diffusion 1.37 → 0.23) at 1.37 s/step after
    compilation.
*   2026-10-27, jobs 7773938 and 7773939 on `nid010100`, two GPUs,
    submitted together by `scripts/submit_nurikabe.sh` into
    `/projects/u6tp/tw1320/diffgemma/trained/nurikabe_lora_20261027T200816Z_bIzX7y`
    (`ckpt_0`, `ckpt_1000` and `ckpt_2000`, 37 GB each, were deleted
    afterwards; logs and wandb files remain). The 2000-step LoRA run
    (rank 8, global batch 2) took 1 h 5 min, with total loss 0.26 at step
    1000 and 0.30 at step 2000. The eval job started on the dependency,
    resolved `latest` to step 2000, and ran `sample_ar_steps64` over 256
    eval examples in 42 min. Results: `nurikabe_accuracy` 0,
    `exact_accuracy` 0, `cell_accuracy` 0.651, `clue_accuracy` 0.246,
    `well_formed` 0.418. Scalars, `nurikabe_boards` images and text samples
    reached TensorBoard and the training job's wandb run
    (`diffgemma-nurikabe/99b3d99675d34525`). In the 10 logged samples, 7
    had 99 to 102 cells instead of 100. Realigning them removed most clue
    mismatches, but no board was valid even after realignment.
*   2026-10-28, allocation 7789101 on `nid010086`, one GH200 through the
    CUDA 13 launcher: both puzzle trajectory notebooks completed with held-out
    record 0 from their config's eval pipeline. Sudoku restored
    `sudoku_lora_7443801/checkpoints/ckpt_2000`; Nurikabe restored
    `nurikabe_lora_20261027T233002Z_NVXIaf/checkpoints/ckpt_7000`. Partial Orbax
    restore resharded the trained network, including LoRA, onto one GPU without
    loading optimizer state. Each run saved 64 states of a 256-token canvas
    (63 updates, temperature 0.7); sampling including compilation took 72.62 s
    and 63.20 s respectively. Sudoku logged a recovered 8 GiB allocator retry.
    Both notebooks also passed CPU-only rerendering. MP4s decoded completely:
    H.264 High, 1920x1080, yuv420p/BT.709, 30 fps, 16.8 s; GIFs are 960x540
    with matching duration. Start/middle/end frames were inspected: malformed
    intermediate fields remain visible and boards have no error colouring or
    clue repairs. Exact HTML text payloads match saved decoded strings.
    Eight trajectory/checkpoint tests, all 17 Sudoku/Nurikabe config tests,
    the launcher's GPU computation check, and notebook schema/syntax/empty
    output checks passed. Artifacts and executed copies are under
    `outputs/{sudoku,nurikabe}_trajectory/`; source notebooks remain output-free.
