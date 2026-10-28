# Fork setup: DiffusionGemma SFT on other machines

This fork of [google-deepmind/gemma](https://github.com/google-deepmind/gemma)
carries our own changes for fine-tuning DiffusionGemma with the
[Hackable Diffusion adapter](gemma/diffusion/hackable_diffusion_adapter/README.md)
(Sudoku first). This file records how to install the fork so that code edits are
live, how to move the work to another machine (including arm64 / GH200
clusters), and what is known not to work. It complements, and where it differs
overrides, the upstream instructions in the adapter README.

`main` mirrors upstream `main`; work happens on the `sudoku-sft` branch. What
has been verified on which machine, and cluster-specific state (drivers,
staged checkpoint paths), is recorded in the adapter's
[AGENT.md](gemma/diffusion/hackable_diffusion_adapter/AGENT.md#-verification-record).

## What this fork changes

| File | Change |
|---|---|
| `pyproject.toml` | `hackable-diffusion` pinned to commit `8581c87` (upstream leaves it unpinned; its HEAD had already moved on 2026-10-02). `etils` pinned to commit `de6d7b2` (the 1.14.0 release fails with jax 0.11, see the [installation notes](#installation-editable-with-uv)). |
| `constraints.txt` | Exact versions of the working x86-64 environment (`uv pip freeze`), used as a constraints file. |
| `excludes-aarch64.txt` | Packages to drop from resolution on arm64 (`tensorflow-cpu`, see [arm64](#arm64--gh200-clusters)). |
| `.gitignore` | Prepared datasets (`*.bagz`, `*.jsonl`, `sudoku.csv`), `xp_dir*/` work directories, `.venv/`, `outputs/` (trajectory videos). |
| `scripts/isambard_cuda13.sh` | Apptainer launcher for the ARM64 checkout's virtual environment; uses the host CUDA driver, or CUDA 13 forward compatibility when the host driver is older. |
| `scripts/check_cuda13.py` | Inspect loaded driver libraries and verify a compiled JAX GPU matrix product. |
| `gemma/diffusion/hackable_diffusion_adapter/trajectory_video.ipynb` | Record one denoising trajectory of the base model and export presentation videos (1080p MP4, GIF, poster, snapshot plot, canvas-only MP4) to `outputs/`. Needs `ipykernel`, `nbclient` and `imageio-ffmpeg` in addition to the base install. The notebook also shows how to register the `diffusiongemma` Jupyter kernel, which runs inside the launcher, for interactive use on a GPU node. |
| `scripts/execute_trajectory_video.py` | Execute that notebook in a fresh kernel through Slurm and the launcher, or only its rendering cells on a CPU with `--render-only`. |
| `FORK_SETUP.md` | This document. |

Everything else is upstream code. Put new work in new files where possible (for
example copy `configs/sft_sudoku.py` to `configs/sft_sudoku_<experiment>.py`
rather than editing it) so that syncing with upstream stays conflict-free.

## Git workflow

```bash
git clone -b sudoku-sft git@github.com:isomorphicdude/gemma.git
cd gemma
git remote add upstream https://github.com/google-deepmind/gemma.git
```

* `origin` is the fork, `upstream` is DeepMind.
* Keep `main` as an untouched mirror of upstream; branch from it for work.
* Sync: `git fetch upstream && git merge upstream/main` (on the work branch).
* Moving between machines is just commit, push, pull. With the editable install
  below no reinstall is needed unless `pyproject.toml` dependencies change.

## Installation (editable, with uv)

The upstream README says `pip install gemma` or `pip install .`. Both copy a
frozen snapshot into `site-packages`, so edits in the checkout are silently
ignored (except when launching from the repo root, where the current directory
shadows the installed copy). Install the checkout in editable mode instead.

Requirements: Python 3.12, [uv](https://docs.astral.sh/uv/) (tested with
0.11.28; the arm64 path needs a version with `--excludes`), an NVIDIA driver new
enough for the CUDA wheels (see below).

The virtual environment lives inside the clone at `.venv/` (git-ignored), so
one directory holds code, environment and data:

```bash
git clone -b sudoku-sft git@github.com:isomorphicdude/gemma.git
cd gemma
uv venv --python 3.12 --prompt diffgemma
source .venv/bin/activate
uv pip install -e . "jax[cuda13]" -c constraints.txt
```

If another environment is active in the shell, uv installs into that one
(`$VIRTUAL_ENV` wins over `./.venv`); either `deactivate` first or add
`--python .venv/bin/python` to the install command.

Verify:

```bash
# Must print a path inside the clone, from any working directory.
python -c "import importlib.util as u; print(u.find_spec('gemma').origin)"
# On a GPU node: must list CudaDevice entries. Restrict to one free GPU and
# skip preallocation so the check does not grab memory on busy GPUs.
CUDA_VISIBLE_DEVICES=0 XLA_PYTHON_CLIENT_PREALLOCATE=false python -c "import jax; print(jax.devices())"
# Loads the Sudoku config without training (from the repo root, CPU only).
JAX_PLATFORMS=cpu python -m gemma.diffusion.hackable_diffusion_adapter.configs.sft_sudoku_test
```

`pytest` is not part of the pinned set; `uv pip install -e ".[dev]"` adds it
if you prefer `pytest gemma/diffusion/hackable_diffusion_adapter/`.

Notes:

* `constraints.txt` reproduces the environment that was built on the x86-64
  workstation: jax/jaxlib 0.11.2 with `jax-cuda13-plugin`, NCCL 2.32, cuDNN
  9.26, kauldron 1.4.4, flax 0.12.10, optax 0.2.8, grain 0.2.18, bagz 0.3.8,
  tensorflow 2.21.0, numpy 2.5.3.
* To refresh the constraints after a deliberate upgrade:
  `uv pip freeze | grep -vE '^(gemma|hackable-diffusion|etils) ' > constraints.txt`
  (the direct-URL entries must stay out, uv rejects a constraint whose URL
  differs from `pyproject.toml`).
* The `hackable-diffusion` and `etils` pins live in `pyproject.toml`, not in
  the constraints file, for the same reason.
* `etils`: the latest release (1.14.0, 2026-03-04) reads the private
  `jax._src.prng.KeyTy`, which jax 0.11 moved to `jax._src.random.prng`. Once
  jax is imported, every `enp.ArraySpec` then raises
  `AttributeError: module 'jax._src' has no attribute 'prng'`; Kauldron hits
  it when it builds `train_ds.element_spec`, before the first training step.
  `pyproject.toml` pins upstream commit `de6d7b2` ("use public JAX APIs for
  detecting PRNG key dtypes", 9 commits after 1.14.0). It still reports
  version 1.14.0, so the `etils==1.14.0` line in `constraints.txt` is
  satisfied. Installing it needs access to github.com. Return to a PyPI
  release once one includes the fix.
* CUDA: the `jax[cuda13]` wheels bundle the CUDA 13 libraries. CUDA 13 needs
  NVIDIA driver 580 or newer on Linux for native compatibility
  ([CUDA release notes](https://docs.nvidia.com/cuda/cuda-toolkit-release-notes/),
  [JAX install docs](https://docs.jax.dev/en/latest/installation.html)).
  Upstream warns that other
  CUDA versions were not tested and can produce NCCL errors.
  Datacenter GPUs can also use CUDA forward compatibility; the validated
  Isambard setup is described [below](#isambard-ai-phase-2-apptainer-launcher).
* If the driver is older than 580, `jax[cuda12]` exists for the same JAX
  version (driver 525+), with the NCCL caveat above.

## arm64 / GH200 clusters

The fork installs on arm64 once `tensorflow-cpu` is excluded; `bagz`
compiles from source; all other 264 packages have Python 3.12 arm64 wheels at
the versions in `constraints.txt`, including the JAX CUDA 13 stack. JAX lists
NVIDIA GPUs on Linux aarch64 as supported.

```bash
# Before installing: load a C++20-capable compiler (bagz builds from source)
# and make sure the node has internet access (bagz downloads Abseil and zstd).
uv pip install -e . "jax[cuda13]" -c constraints.txt --excludes excludes-aarch64.txt
```

Why each deviation:

* **`tensorflow-cpu`** has no arm64 build in any version, and both `gemma` and
  `kauldron` 1.4.4 require it, so a plain install fails at resolution.
  Excluding it loses nothing: the full `tensorflow` 2.21.0 wheel exists for
  arm64 and is pulled in anyway by `tensorflow-text`. Plain pip has no
  equivalent of `--excludes`; use uv.
* **`opencv-python`** (required by `kauldron`) links X11/xcb/GL system
  libraries (`libxcb.so.1`, `libGL.so.1`, ...) that the CUDA base container
  lacks, so `import cv2` fails there. TFDS imports it optionally and catches
  the error, but OpenCV's loader leaves `site-packages/cv2` on `sys.path` after
  the failure. Grain's spawned data workers inherit that path, `import typing`
  resolves to `cv2/typing`, and every worker dies on `libxcb.so.1`. Nothing in
  this stack uses `cv2`, so it is excluded. If an existing venv has it:
  `uv pip uninstall opencv-python`.
* **`bagz`** 0.3.8 ships Linux wheels for x86-64 only. On arm64 pip/uv build it
  from the sdist: CMake 3.28+ (fetched automatically if the system one is
  older), a C++20 compiler, and internet access during the build. Nothing in
  it is x86-specific and upstream ships macOS arm64 wheels. If a build fails:
  `bagz` is only imported by `data/sudoku/sudoku_data.py` (reader) and
  `data/sudoku/convert_sudoku.py` (writer). The on-disk format is simple
  (each record is a zstd frame, followed by a tail table of little-endian
  uint64 end offsets, the last 8 bytes giving the table start; verified on the
  prepared eval file: 900,000 records), so a small pure-Python reader using the
  `zstandard` package can replace `bagz.Reader` for training.
* **glibc**: the arm64 wheels need glibc 2.28 or newer (`ldd --version`).

Checks to make on a new cluster:

* A driver compatible with the CUDA wheels, either natively or through a
  validated CUDA forward compatibility setup.
* Whether the compute nodes have internet; if not, see
  [offline provisioning](#base-checkpoint-and-tokenizer).
* The number of GPUs per node, which constrains the batch size (see
  [training](#training)).
* Memory: a GH200 GPU has 96 GB; the base weights are 37.6 GiB on disk and the
  model defaults to bfloat16. Upstream recommends at least two A100s for the
  LoRA config, so plan for two or more GPUs until a run on one GPU is confirmed.

## Isambard-AI Phase 2: Apptainer launcher

Use the project launcher with Apptainer 1.4.1. It follows
[Isambard's GPU container guidance](https://docs.isambard.ac.uk/user-documentation/guides/containers/apptainer/#using-gpus-with-apptainer)
and the
[CUDA forward compatibility guide](https://docs.isambard.ac.uk/user-documentation/guides/gpus_and_cuda/#cuda-forward-compatibility).
The base image is NVIDIA `cuda:13.0.0-base-ubuntu24.04`, pinned by its ARM64
manifest digest. It supplies R580 CUDA compatibility libraries and an Ubuntu
runtime; Python and Gemma remain in the editable checkout's existing `.venv`.
An R580 driver (the image's 580.65.06 compatibility driver, or an R580 host
driver) runs the newer wheel runtimes through
[CUDA minor version compatibility](https://docs.nvidia.com/deploy/cuda-compatibility/minor-version-compatibility.html).
Do not move to a newer image tag without a GPU workload check; the CUDA 13.4.1
image already failed on this cluster (see the verification record).

Download the image once on a login node, then check the actual JAX environment
through Slurm on a GPU. Downloads from compute nodes can be much slower; image
compression is limited to two CPU threads on the login node.

```bash
cd /path/to/gemma
scripts/isambard_cuda13.sh --pull
srun --nodes=1 --gpus=1 --ntasks=1 --time=00:05:00 \
    scripts/isambard_cuda13.sh --check
```

The SIF is stored in `.apptainer/` (git-ignored). To store it in project storage,
set `GEMMA_CUDA13_IMAGE=/projects/<project>/<user>/containers/cuda-13.0.0-arm64.sif`
before both commands. The launcher binds the checkout, the home directory
(including uv's Python installation), and `/projects`, and always runs from the
repository root so relative dataset paths work.

`--nv` exposes the GPU and binds host NVIDIA libraries into
`/.singularity.d/libs`. The launcher uses `--cleanenv` to exclude host module and
Python paths. It then compares the host kernel driver version (from
`/proc/driver/nvidia/version`) with the image's compatibility driver. Only when
the compatibility driver is newer does it set the container's `LD_LIBRARY_PATH`
to `/usr/local/cuda/compat:/.singularity.d/libs`; otherwise it sets
`/.singularity.d/libs` alone and the host driver is used. A compatibility
driver older than the kernel driver fails `cuInit` with
`CUDA_ERROR_SYSTEM_DRIVER_MISMATCH` (803), and JAX then silently falls back to
CPU. The image's CUDA runtime library directory is excluded so the pinned
`.venv` wheels supply their own newer CUDA runtime/math libraries. It
forwards Slurm's `CUDA_VISIBLE_DEVICES`, the documented XLA/NCCL training
settings, and uses `.venv/bin` first in `PATH`. It does not add a global library-path override
to the shell profile. The check forces the CUDA backend and disables GPU memory
preallocation, prints the loaded `libcuda`, checks initialization and the matrix
result, and runs an NCCL all-reduce when more than one GPU is allocated.
Results of these checks on the cluster, and its driver history, are in the
[verification record](gemma/diffusion/hackable_diffusion_adapter/AGENT.md#-verification-record).

Launch Sudoku SFT inside a Slurm job with the same tuning flags as above:

```bash
XLA_FLAGS="--xla_disable_hlo_passes=constant_folding" \
NCCL_ALGO=Ring NCCL_PROTO=LL128 NCCL_NVLS_ENABLE=0 NCCL_CUMEM_ENABLE=0 \
srun --nodes=1 --gpus=2 --ntasks=1 \
    scripts/isambard_cuda13.sh python -m kauldron.main \
    --cfg=gemma/diffusion/hackable_diffusion_adapter/configs/sft_sudoku.py \
    --cfg.workdir=/projects/<project>/<user>/xp_sudoku_lora
```

For an interactive session, use
`srun --nodes=1 --gpus=1 --ntasks=1 --pty scripts/isambard_cuda13.sh`.
Additional application variables can be passed using `APPTAINERENV_<NAME>`;
Apptainer's `--no-eval` keeps their values literal. This launcher is for a single
node; multi-node transport and a full training run require separate validation.
The compatibility driver is only a fallback for nodes whose driver is older
than 580.65.06.
[NVIDIA's current forward compatibility matrix](https://docs.nvidia.com/deploy/cuda-compatibility/forward-compatibility.html)
does not list R565 as a supported target, even though Isambard documents this
approach. Repeat the `--check` workload check when changing the image or the
CUDA wheels, and after a cluster driver update.

## Data

`prepare_sudoku_dataset.sh` (Kaggle token required, see the adapter README)
produces, in `gemma/diffusion/hackable_diffusion_adapter/data/sudoku/`:

| File | Size | Needed for |
|---|---|---|
| `sudoku.csv` | 1.4 GB | conversion only |
| `sudoku_train.bagz` | 1.6 GB, 8,100,000 records | training |
| `sudoku_eval.bagz` | 187 MB, 900,000 records | evaluation |

These are git-ignored (GitHub rejects files above 100 MB). To move them to
another machine, copy the two `.bagz` files into the same directory, for
example:

```bash
rsync -avP gemma/diffusion/hackable_diffusion_adapter/data/sudoku/*.bagz \
    other-machine:/path/to/gemma/gemma/diffusion/hackable_diffusion_adapter/data/sudoku/
```

The files are architecture-independent (little-endian on both x86-64 and
arm64). The configs reference them with paths relative to the repo root, so
training must be launched from the repo root (or override
`--cfg.train_ds.bagz_path` and `--cfg.eval_ds.bagz_path`).

## Base checkpoint and tokenizer

Both are fetched from a public GCS bucket at run time; no credentials needed,
but the node needs internet access.

* Checkpoint: `gs://gemma-data/checkpoints/diffusiongemma-26B-A4B-it`
  (37.6 GiB, 32 objects), loaded by `cfg.init_transform` at start-up.
* Tokenizer: `gs://gemma-data/tokenizers/tokenizer_gemma4.model` (4.7 MB),
  loaded on first use. `gemma.gm.utils._file_cache` looks for a local copy at
  `$GEMMA_CACHE_DIR/tokenizer/tokenizer_gemma4.model` (default
  `~/.gemma/tokenizer/`) before falling back to GCS.

Offline provisioning (download on a login node, then point the run at the
copies):

```bash
mkdir -p ~/.gemma/tokenizer
curl -o ~/.gemma/tokenizer/tokenizer_gemma4.model \
    https://storage.googleapis.com/gemma-data/tokenizers/tokenizer_gemma4.model
gcloud storage rsync -r gs://gemma-data/checkpoints/diffusiongemma-26B-A4B-it \
    /path/to/ckpts/diffusiongemma-26B-A4B-it
# then add to the training command:
#   --cfg.aux.checkpoint_path=/path/to/ckpts/diffusiongemma-26B-A4B-it
```

Verify the copy before training (no GPU needed):

```bash
du -sh /path/to/ckpts/diffusiongemma-26B-A4B-it  # expect ~37.6 GiB
python -c "import orbax.checkpoint as ocp; \
    print(ocp.PyTreeCheckpointer().metadata('/path/to/ckpts/diffusiongemma-26B-A4B-it'))"
```

Override the checkpoint path with `--cfg.aux.checkpoint_path=...` ( `sft_sudoku.py` stores the default in the plain-string field `cfg.aux.checkpoint_path` and wires `init_transform.path` to it via `cfg.ref)`. 

> [!Note]
> `sft_pubmedqa.py` and `sft_sudoku_full.py` still hardcode
`CHECKPOINT_PATH` and lack the `aux.checkpoint_path` field.

The copies staged on Isambard Phase 2 are listed in the adapter's
[AGENT.md](gemma/diffusion/hackable_diffusion_adapter/AGENT.md#6-isambard-staged-checkpoint-and-tokenizer).

## Training

Current Sudoku config (`configs/sft_sudoku.py`):

* DiffusionGemma 26B-A4B with rank-8 LoRA on all linear layers; only LoRA
  parameters are optimised.
* One canvas of 256 tokens, prompt padded to 256; uniform categorical
  corruption with a linear schedule; loss = denoiser loss + encoder AR loss.
* 2,000 steps, batch size 2 (global, across all visible GPUs; the upstream README
  says 8), Adam with peak LR 1.5e-4, 100 warm-up steps, cosine decay.
* Checkpoint every 1,000 steps; fused LoRA weights written to
  `<workdir>/gemma_like_params/fused_lora_params_<step>`.
* No accuracy evaluation during training: the sampling evaluators are built in
  the config but only attached by `eval_main.py`, which runs offline.

Launch from the repo root, with the upstream environment variables:

```bash
cd /path/to/gemma    # repo root
env XLA_FLAGS="--xla_disable_hlo_passes=constant_folding" \
    NCCL_ALGO="Ring" NCCL_PROTO="LL128" NCCL_NVLS_ENABLE="0" NCCL_CUMEM_ENABLE="0" \
    python -m kauldron.main \
    --cfg=gemma/diffusion/hackable_diffusion_adapter/configs/sft_sudoku.py \
    --cfg.workdir=/path/outside/the/repo/xp_sudoku_lora
```

* Batch size vs GPUs: Kauldron shards the batch along its first dimension
  across every visible device, so the number of visible GPUs must divide the
  batch size. With the default batch size 2, restrict to one or two GPUs
  (`CUDA_VISIBLE_DEVICES=0,1`) or raise `--cfg.train_ds.batch_size`. Derived
  from Kauldron's default sharding, not yet observed in a run.
* `--cfg.workdir` inside the repo works too (`xp_dir*/` is git-ignored), but
  keeping it on scratch storage is cleaner.
* `configs/sft_sudoku_test.py` asserts batch size 2 and the default bagz paths;
  update it if you change the config.

## Evaluation

Offline, from the repo root (see the adapter README for the flag table):

```bash
env XLA_FLAGS="--xla_disable_hlo_passes=constant_folding" \
    XLA_PYTHON_CLIENT_PREALLOCATE="false" TF_FORCE_GPU_ALLOW_GROWTH="true" \
    python -m gemma.diffusion.hackable_diffusion_adapter.eval_main \
    --cfg=gemma/diffusion/hackable_diffusion_adapter/configs/sft_sudoku.py \
    --task=sudoku --step=1000 --eval_names=sample_ar_steps64 \
    --cfg.workdir=/path/outside/the/repo/xp_sudoku_lora \
    --cfg.eval_ds.batch_size=2 --cfg.aux.eval_num_batches=2 --cfg.aux.num_canvases=2
```

Metrics land in TensorBoard event files under the work directory.

## Known limitations

1. No training run has completed from this fork yet on any machine. The
   install, GPU, data and config checks that have passed, and the failed
   training attempts, are listed in the
   [verification record](gemma/diffusion/hackable_diffusion_adapter/AGENT.md#-verification-record).
2. Other ARM64 installations still need a source build of `bagz`; see the
   arm64 section for the fallback.
3. CUDA 13 wheels need a compatible driver or a validated CUDA forward
   compatibility setup. Isambard-AI Phase 2 uses the Apptainer launcher above;
   `jax[cuda12]` is an alternative that remains untested with this adapter and
   carries the upstream NCCL warning.
4. Start-up needs internet for the checkpoint and tokenizer unless they are
   provisioned locally as described above. The Isambard Phase 2 copies, and
   the settings runs there must pass, are listed in AGENT.md.
5. `hackable-diffusion` is pinned to `8581c87`. Upstream HEAD (`d53bfcb`,
   2026-10-02) only adds `noise_frac` to the Gaussian DDIM step, which this
   adapter does not use, but future upstream changes to the adapter may
   require moving the pin.
6. On x86-64 both `tensorflow` and `tensorflow-cpu` get installed (upstream
   dependency quirk); harmless, and the arm64 path simply drops the latter.
7. `promise` (a `tensorflow-datasets` dependency) has no wheel on any platform
   and is built from its pure-Python sdist; this is the same on x86-64.
8. The upstream README's batch size (8) and the config's (2) disagree; the
   config is what runs.
