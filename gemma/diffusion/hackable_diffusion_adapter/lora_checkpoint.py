"""Export and reload adapter-only DiffusionGemma checkpoints.

The portable bundle contains an Orbax LoRA checkpoint and model layout metadata.
No base model tensors, optimizer state, or machine-specific paths are exported.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
import tempfile


FORMAT = "gemma-diffusion-lora"


def _validate_adapter(params, rank):
  """Reject empty adapters, base weights, and incomplete/mismatched A/B pairs."""
  from flax import traverse_util  # pylint: disable=g-import-not-at-top

  flat = traverse_util.flatten_dict(params)
  if not flat:
    raise ValueError("No LoRA adapter tensors found")
  pairs = {}
  for path, value in flat.items():
    if len(path) < 2 or path[-2] != "lora" or path[-1] not in ("a", "b"):
      raise ValueError(f"Unexpected non-adapter tensor: {'/'.join(path)}")
    axis = -1 if path[-1] == "a" else 0
    if not value.shape or value.shape[axis] != rank:
      raise ValueError(f"LoRA rank mismatch at {'/'.join(path)}")
    pairs.setdefault(path[:-1], set()).add(path[-1])
  if any(keys != {"a", "b"} for keys in pairs.values()):
    raise ValueError("Incomplete LoRA A/B pair")
  return flat


def export_adapter(checkpoint, output_dir, *, task="nurikabe"):
  """Extract only LoRA tensors from a completed Kauldron training checkpoint.

  Reuses the repo's partial network restore and adapter checkpoint formatter.
  The training config supplies rank/targets; base specs preserve the parameter
  names and dtypes used in training, including Gemma's shared-scope layout.
  Existing output directories are never overwritten.
  """
  from flax import traverse_util  # pylint: disable=g-import-not-at-top
  from gemma import diffusion, peft
  from gemma.diffusion.hackable_diffusion_adapter import puzzle_trajectory as pt
  from gemma.diffusion.hackable_diffusion_adapter.hd import gemma_checkpointer
  import jax
  import orbax.checkpoint as ocp

  checkpoint = pt.latest_checkpoint(task, checkpoint=checkpoint)
  output_dir = Path(output_dir).expanduser().resolve()
  if output_dir.exists():
    raise FileExistsError(f"Adapter output already exists: {output_dir}")
  run_dir = checkpoint.parent.parent
  training = json.loads((run_dir / "config.json").read_text())
  network = training["model"]["gemma_network"]
  if (not network["__qualname__"].endswith(":LoRA")
      or training["optimizer"].get("mask", {}).get("0") != "lora"):
    raise ValueError("Expected a training run that updates only LoRA parameters")
  rank = int(network["rank"])
  with ocp.PyTreeCheckpointer() as checkpointer:
    metadata = checkpointer.metadata(checkpoint / "default").item_metadata.tree
    base = peft.split_params(metadata["params"]["gemma_network"]["gemma_model"]).original
  base_specs = {
      path: {"shape": list(value.shape), "dtype": str(value.dtype)}
      for path, value in traverse_util.flatten_dict(base, sep="/").items()
  }
  adapter = pt.restore_network_params(
      checkpoint, device=jax.devices("cpu")[0], lora_only=True)["gemma_model"]
  flat = _validate_adapter(adapter, rank)
  step = int(checkpoint.name.removeprefix("ckpt_"))
  config = {
      "format": FORMAT,
      "format_version": 1,
      "task": task,
      "base_model": "diffusiongemma-26B-A4B-it",
      "base_checkpoint": str(diffusion.CheckpointPath.DIFFUSIONGEMMA_26B_A4B_IT),
      "checkpoint_step": step,
      "source_run": run_dir.name,
      "rank": rank,
      "target_modules": network.get("target_modules"),
      "weights": f"lora_params_{step}",
      "num_tensors": len(flat),
      "num_parameters": sum(math.prod(value.shape) for value in flat.values()),
      "base_parameter_specs": base_specs,
  }
  output_dir.parent.mkdir(parents=True, exist_ok=True)
  with tempfile.TemporaryDirectory(
      prefix=f".{output_dir.name}-", dir=output_dir.parent) as temp:
    staging = Path(temp)
    gemma_checkpointer.save_params_into_original_and_lora_params(
        adapter, step, staging, write_original_params=False,
        write_lora_params=True, write_fused_lora_params=False)
    (staging / "adapter_config.json").write_text(json.dumps(config, indent=2) + "\n")
    (staging / "README.md").write_text(
        "# Nurikabe LoRA adapter\n\n"
        f"Rank-{rank} LoRA adapter for DiffusionGemma 26B-A4B IT, step {step}.\n"
        "This repository-native JAX/Orbax bundle contains adapter tensors only.\n"
        "The base checkpoint must be obtained separately.\n\n"
        "Use `nurikabe_trajectory_video_github.ipynb` in the Gemma checkout to\n"
        "load the base weights, merge this adapter, and fuse it for sampling.\n")
    staging.rename(output_dir)
  return config


def load_adapter(directory, *, device=None):
  """Load a portable bundle and restore its adapter tensors on one device."""
  import jax  # pylint: disable=g-import-not-at-top
  import orbax.checkpoint as ocp

  directory = Path(directory).expanduser().resolve()
  config = json.loads((directory / "adapter_config.json").read_text())
  if config.get("format") != FORMAT or config.get("format_version") != 1:
    raise ValueError("Unsupported LoRA adapter format")
  device = device or jax.devices()[0]
  sharding = jax.sharding.SingleDeviceSharding(device)
  with ocp.PyTreeCheckpointer() as checkpointer:
    path = directory / config["weights"]
    metadata = checkpointer.metadata(path).item_metadata.tree
    _validate_adapter(metadata, config["rank"])
    item = jax.tree.map(
        lambda m: jax.ShapeDtypeStruct(m.shape, m.dtype, sharding=sharding), metadata)
    args = jax.tree.map(lambda m: ocp.ArrayRestoreArgs(
        restore_type=jax.Array, dtype=m.dtype, global_shape=m.shape,
        sharding=sharding), item)
    params = checkpointer.restore(
        path, args=ocp.args.PyTreeRestore(item=item, restore_args=args))
  return config, params


def align_base_params(base_params, adapter_config):
  """Match a base checkpoint to the names, shapes, and dtypes used in training.

  Uses the training loader's existing '/w' remapping and rejects missing or
  incompatible base tensors before the adapter is fused.
  """
  from flax import traverse_util  # pylint: disable=g-import-not-at-top
  from gemma.diffusion.hackable_diffusion_adapter.hd import gemma_checkpointer
  import jax
  import jax.numpy as jnp

  specs = adapter_config["base_parameter_specs"]
  target = {path: jax.ShapeDtypeStruct(value["shape"], jnp.dtype(value["dtype"]))
            for path, value in specs.items()}
  flat = gemma_checkpointer._remap_and_match_params(  # pylint: disable=protected-access
      target, traverse_util.flatten_dict(base_params, sep="/"))
  for path, value in flat.items():
    spec = specs[path]
    if tuple(value.shape) != tuple(spec["shape"]):
      raise ValueError(f"Base parameter shape mismatch at {path}")
    flat[path] = value.astype(jnp.dtype(spec["dtype"]))
  return traverse_util.unflatten_dict(flat, sep="/")
