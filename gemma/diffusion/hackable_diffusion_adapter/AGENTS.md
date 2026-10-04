# Adapter agent instructions

Read and follow [AGENT.md](AGENT.md) before working in this directory or its
subdirectories. It contains the adapter's setup, testing, training,
evaluation, and code modification rules. Also read the repository's
[FORK_SETUP.md](../../../FORK_SETUP.md) for the fork workflow and provisioning.

On Isambard-AI Phase 2:

* Run commands from the repository root, which contains `pyproject.toml`.
* Build/download the pinned image once on a login node with
  `scripts/isambard_cuda13.sh --pull`; reuse it in compute jobs.
* Run GPU commands through Slurm and `scripts/isambard_cuda13.sh`, using the
  editable `.venv` and existing CUDA 13 constraints. Preserve Slurm's GPU
  selection and keep compatibility library paths inside the launcher.
* Use the playbook's `--check` and Sudoku config test commands for validation.
  The verified scope covers GPU computation, a two-GPU all-reduce, and nine
  config tests; full training and multi-node jobs remain untested.
