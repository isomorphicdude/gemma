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
    *   [`sft_sudoku.py`](gemma/diffusion/hackable_diffusion_adapter/configs/sft_sudoku.py): LoRA-based SFT training for the Sudoku puzzle solving task.
    *   [`sft_sudoku_full.py`](gemma/diffusion/hackable_diffusion_adapter/configs/sft_sudoku_full.py): Full weight SFT training for the Sudoku puzzle solving task (no LoRA).
    *   [`sft_pubmedqa.py`](gemma/diffusion/hackable_diffusion_adapter/configs/sft_pubmedqa.py): LoRA-based SFT training for the PubMedQA long-answer task.
*   **`data/`**: Dataset loading, custom pipelines, and preprocessing transforms.
    *   [`data.py`](gemma/diffusion/hackable_diffusion_adapter/data/data.py): Common transforms (e.g. `CanvasChunker` for localized diffusion).
*   **`hd/`**: Core modeling, network layers, and state handling.
    *   [`sft_model.py`](gemma/diffusion/hackable_diffusion_adapter/hd/sft_model.py): Core `SFTDiffusion` class managing the hybrid AR prefill and localized diffusion denoising steps.
    *   [`lora.py`](gemma/diffusion/hackable_diffusion_adapter/hd/lora.py): PEFT LoRA wrappers.
    *   [`mask_helpers.py`](gemma/diffusion/hackable_diffusion_adapter/hd/mask_helpers.py): Right-pad causal/block masks and cursor tracking.
*   **`eval/`**: Custom evaluation metrics designed to avoid TPU/GPU OOM issues.
    *   [`sudoku_eval.py`](gemma/diffusion/hackable_diffusion_adapter/eval/sudoku_eval.py): Unified host-side `SudokuAllMetrics` evaluation.

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
*   NVIDIA driver 580+ on the compute nodes (check `nvidia-smi`) — required
    by the CUDA 13 wheels.
*   glibc 2.28 or newer (`ldd --version`).
*   A C++20-capable compiler loaded (e.g. `module load gcc`) and internet
    access during installation: `bagz` has no arm64 wheel and builds from
    source, downloading Abseil and zstd.

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

### 4. Verify

```bash
# Must print a path inside the clone, from any working directory.
python -c "import importlib.util as u; print(u.find_spec('gemma').origin)"
# On a GPU node: must list CudaDevice entries. Restrict to one free GPU and
# skip preallocation so the check does not grab memory on busy GPUs.
CUDA_VISIBLE_DEVICES=0 XLA_PYTHON_CLIENT_PREALLOCATE=false python -c "import jax; print(jax.devices())"
# Loads the Sudoku config without training (from the repo root, CPU only).
JAX_PLATFORMS=cpu python -m gemma.diffusion.hackable_diffusion_adapter.configs.sft_sudoku_test
```

> **CUDA Version Constraint**: If configuring JAX for GPU, you must use
> **CUDA 13** (driver 580+; the wheels bundle the CUDA 13 libraries).
> Mixing CUDA 12 packages (such as `jax-cuda12-plugin` or
> `nvidia-nccl-cu12`) will trigger PJRT initialization crashes and silent
> NCCL corruption. If the driver is older than 580, `jax[cuda12]` exists for
> the same JAX version (driver 525+), untested with this adapter.

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

`pytest` is not part of the pinned environment; install it first with
`uv pip install -e ".[dev]"`.

To verify the JAX layers, data pipelines, and sampling routines, run:

```bash
pytest gemma/diffusion/hackable_diffusion_adapter/
```

Or run individual tests:
```bash
pytest gemma/diffusion/hackable_diffusion_adapter/hd/lora_test.py
```

---

## 🚀 Launching SFT Training

Always use the standard Kauldron CLI command. Use the following env variables
to prevent JIT compilation OOMs and NCCL communication hangs:

Launches should be run from the parent dir of the gemma directory.

#### PubMedQA

```bash
env XLA_FLAGS="--xla_disable_hlo_passes=constant_folding" \
    NCCL_ALGO="Ring" \
    NCCL_PROTO="LL128" \
    NCCL_NVLS_ENABLE="0" \
    NCCL_CUMEM_ENABLE="0" \
    python3 -m kauldron.main \
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
    python3 -m kauldron.main \
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
    python3 -m kauldron.main \
  --cfg=gemma/diffusion/hackable_diffusion_adapter/configs/sft_sudoku_full.py \
  --cfg.workdir=$(pwd)/xp_dir
```

---

## 📊 Offline Evaluation

Evaluation is run **offline** — it is a separate step from training. The
`eval_main` binary loads a saved checkpoint, runs AR diffusion sampling
on the eval dataset, and reports task-specific metrics.

### Running an Eval Job

From the parent dir of the gemma directory:

```bash
env XLA_FLAGS="--xla_disable_hlo_passes=constant_folding" \
    XLA_PYTHON_CLIENT_PREALLOCATE="false" \
    TF_FORCE_GPU_ALLOW_GROWTH="true" \
    python3 -m gemma.diffusion.hackable_diffusion_adapter.eval_main \
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
[`ar_eval.make_ar_evals`](gemma/diffusion/hackable_diffusion_adapter/eval/ar_eval.py).
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
