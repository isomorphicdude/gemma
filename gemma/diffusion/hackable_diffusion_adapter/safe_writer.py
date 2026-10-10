# Copyright 2026 DeepMind Technologies Limited.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Safe metric writer that avoids NCCL crashes on multi-GPU.

Overrides ``write_param_overview`` to count parameters without
calling ``jax.device_get()`` on sharded arrays.
"""

from typing import Literal, Mapping, Any
import functools
import json
import hashlib
import dataclasses
from etils import epath
from kauldron.train import metric_writer
from kauldron.utils.status_utils import status

# --- Patch B: Safe _convert_leaf using single-shard reads ---
# Never call jax.device_get() on sharded arrays.  Always read from a
# single local shard to avoid triggering NCCL AllGather.
import jax
import numpy as np
import kauldron.train.auxiliaries as aux


def _convert_leaf(leaf):
    if isinstance(leaf, jax.Array):
        shard_data = leaf.addressable_shards[0].data
        return np.asarray(shard_data)
    return leaf


aux._convert_leaf = _convert_leaf

# --- Patch C: Safe _compute_metric with synchronization ---
# Force jax.block_until_ready after metric computation to prevent
# concurrent NCCL operations that corrupt the communicator.

_orig_compute = aux.AuxiliariesState.compute


def _safe_compute(self, *args, **kwargs):
    result = _orig_compute(self, *args, **kwargs)
    jax.block_until_ready(jax.tree_util.tree_leaves(result))
    return result


aux.AuxiliariesState.compute = _safe_compute


@dataclasses.dataclass(frozen=True, eq=True, kw_only=True)
class SafeMetricWriter(metric_writer.KDMetricWriter):
    """Metric writer that avoids NCCL crashes on multi-GPU.

    Overrides ``write_param_overview`` to count parameters without
    calling ``jax.device_get()`` on sharded arrays.
    """

    def write_param_overview(self, step: int, params) -> None:
        num_parameters = metric_writer.parameter_overview.count_parameters(params)
        self.write_summaries(step=step, values={"num_params": num_parameters})


@dataclasses.dataclass(frozen=True, eq=True, kw_only=True)
class SafeWandbWriter(SafeMetricWriter):
    """Safe wrapper around wandb for distributed training.

    Ensures safe initialization and logging across multiple processes.
    """

    wbproj: str = "gemma-diffusion-finetune"
    entity: str | None = None
    runname: str | None = None
    mode: Literal["online", "offline", "disabled", "shared"] = "online"
    run_name_parts: Mapping[str, Any] | None = None

    @functools.cached_property
    def _wandb(self):
        if not status.is_lead_host:
            return None
        import wandb  # pylint: disable=g-import-not-at-top

        if wandb.run is None:
            workdir = epath.Path(self.workdir)
            run_id = hashlib.sha1(f"{workdir}".encode()).hexdigest()[:16]
            wandb.init(
                project=self.wbproj,
                entity=self.entity,
                name=f"{self._run_name}",
                id=run_id,
                resume="allow",
                mode=self.mode,
                dir=str(workdir),
            )
            wandb.define_metric("step")
            wandb.define_metric("*", step_metric="step")
        return wandb

    def write_scalars(self, step, scalars) -> None:
        super().write_scalars(step, scalars)
        if self._wandb is not None and scalars:
            self._wandb.log(
                {"step": step}
                | {
                    f"{self.collection}/{k}": np.asarray(v).item()
                    for k, v in scalars.items()
                }
            )
        #TODO: add logging of sudoku grid images

    def write_config(self, config) -> None:
        super().write_config(config)
        # the parent does not export config but we upload it
        if self._wandb is not None and config is not None:
            self._wandb.config.update(
                json.loads(config.to_json()), allow_val_change=True
            )

    @functools.cached_property
    def _run_name(self) -> str:
        if self.runname:
            return self.runname
        if self.run_name_parts:
            return "-".join(f"{k}{v}" for k, v in self.run_name_parts.items())
        return epath.Path(self.workdir).name
