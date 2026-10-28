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

"""Nurikabe evaluation utilities: scoring and accuracy metric.

Boards are 100 space-separated cells in row-major order: ``?`` for an unknown
puzzle cell, ``#`` for sea (black) and the island size for an island cell
(white). A shading is scored as solved when ``nurikabe_verify.verify`` accepts
it; the island-size numbers are only checked by the exact-match score.
"""

from __future__ import annotations

from typing import Any
import dataclasses

import flax.struct
from gemma.diffusion.hackable_diffusion_adapter.eval import base_metric
import jax.numpy as jnp
from kauldron import kd
import numpy as np

from gemma.diffusion.hackable_diffusion_adapter.eval import nurikabe_verify
from gemma.diffusion.hackable_diffusion_adapter.eval import nurikabe_viz

################################################################################
# MARK: Constants
################################################################################

_NUM_CELLS = nurikabe_viz.NUM_CELLS


################################################################################
# MARK: Extraction
################################################################################


def extract_nurikabe_cells(text: str) -> list[str] | None:
    """Extract the first 100 cell tokens from generated text.

    Cell tokens are ``#``, ``?`` and whole numbers (so ``10`` is one cell); any
    other text, such as turn markers, is ignored.

    Args:
      text: The raw text produced by the model.

    Returns:
      A list of exactly 100 cell tokens, or ``None`` if the text has fewer.
    """
    cells = nurikabe_viz.CELL_RE.findall(text)
    if len(cells) < _NUM_CELLS:
        return None
    return cells[:_NUM_CELLS]


def _reference_cells(s: str) -> list[str] | None:
    """Cells of a ground-truth or puzzle string, or ``None`` if not 100 cells."""
    cells = nurikabe_viz.CELL_RE.findall(s)
    return cells if len(cells) == _NUM_CELLS else None


################################################################################
# MARK: Scoring helpers
################################################################################


def valid_solution(generated: str, ground_truth: str, puzzle: str) -> float:
    """Check that the generated shading is a valid Nurikabe solution.

    Every cell must be ``#`` or an island size, and the shading must pass
    ``nurikabe_verify.verify`` for the puzzle's clues. Island-size numbers are
    not checked. The dataset's solutions are unique, so a valid shading also
    matches the ground-truth shading.

    Args:
      generated: The generated text.
      ground_truth: The ground-truth solution string (unused, kept for API
        consistency with the other scoring helpers).
      puzzle: The original puzzle string (``?`` = unknown cell).

    Returns:
      ``1.0`` if the shading is valid, ``0.0`` otherwise.
    """
    del ground_truth  # Unused — validity depends only on the clues.
    gen = extract_nurikabe_cells(generated)
    puz = _reference_cells(puzzle)
    if gen is None or puz is None:
        return 0.0
    if any(c != "#" and c not in nurikabe_viz.ISLAND_SIZES for c in gen):
        return 0.0
    clues = nurikabe_viz.clue_grid(puz)
    black = nurikabe_viz.shading_grid(gen).astype(int)
    return 0.0 if nurikabe_verify.verify(clues.tolist(), black.tolist()) else 1.0


def exact_accuracy(generated: str, ground_truth: str, puzzle: str) -> float:
    """Check that all 100 generated cells, island sizes included, match the target.

    Args:
      generated: The generated text.
      ground_truth: The ground-truth solution string.
      puzzle: The original puzzle string (unused, kept for API consistency with
        the other scoring helpers).

    Returns:
      ``1.0`` if every cell matches the ground truth, ``0.0`` otherwise.
    """
    del puzzle  # Unused — exact accuracy compares full grids.
    gen = extract_nurikabe_cells(generated)
    gt = _reference_cells(ground_truth)
    if gen is None or gt is None:
        return 0.0
    return 1.0 if gen == gt else 0.0


def cell_accuracy(generated: str, ground_truth: str, puzzle: str) -> float:
    """Fraction of unknown (``?``) puzzle cells shaded as in the ground truth.

    Only sea versus island is compared, not the island-size numbers.

    Args:
      generated: The generated text.
      ground_truth: The ground-truth solution string.
      puzzle: The original puzzle string (``?`` = unknown cell).

    Returns:
      A float in ``[0, 1]``. Returns ``0.0`` if the inputs are invalid or the
      cells could not be extracted.
    """
    gen = extract_nurikabe_cells(generated)
    gt = _reference_cells(ground_truth)
    puz = _reference_cells(puzzle)
    if gen is None or gt is None or puz is None:
        return 0.0
    unknown = [j for j in range(_NUM_CELLS) if puz[j] == "?"]
    if not unknown:
        return 1.0
    correct = sum(1 for j in unknown if (gen[j] == "#") == (gt[j] == "#"))
    return correct / len(unknown)


def clue_accuracy(generated: str, ground_truth: str, puzzle: str) -> float:
    """Check that the model kept every clue cell's number unchanged.

    Does **not** verify the unknown cells.

    Args:
      generated: The generated text.
      ground_truth: The ground-truth solution string (unused, kept for API
        consistency with the other scoring helpers).
      puzzle: The original puzzle string (``?`` = unknown cell).

    Returns:
      ``1.0`` if all clues are preserved, ``0.0`` otherwise.
    """
    del ground_truth  # Unused — only puzzle cells matter.
    gen = extract_nurikabe_cells(generated)
    puz = _reference_cells(puzzle)
    if gen is None or puz is None:
        return 0.0
    preserved = all(g == p for g, p in zip(gen, puz) if p != "?")
    return 1.0 if preserved else 0.0


def well_formed(generated: str, ground_truth: str, puzzle: str) -> float:
    """Check that the output is exactly 100 cells, each ``#`` or an island size.

    Args:
      generated: The generated text.
      ground_truth: Unused, kept for API consistency.
      puzzle: Unused, kept for API consistency.

    Returns:
      ``1.0`` if the output is well formed, ``0.0`` otherwise.
    """
    del ground_truth, puzzle  # Unused — only the output's format matters.
    return 1.0 if nurikabe_viz.parse_cells(generated)[1] else 0.0


################################################################################
# MARK: Metric
################################################################################

_ALL_SCORING_FNS = (
    valid_solution,
    exact_accuracy,
    cell_accuracy,
    clue_accuracy,
    well_formed,
)
_SCORING_NAMES = (
    "accuracy",
    "exact_accuracy",
    "cell_accuracy",
    "clue_accuracy",
    "well_formed",
)


@dataclasses.dataclass(kw_only=True, frozen=True)
class NurikabeAllMetrics(base_metric.BaseTextMetric):
    """Compute all Nurikabe scores and the sample count.

    Stores raw token arrays in the metric state and performs all detokenization
    and scoring on the host in ``compute()``.

    Reports 6 sub-metrics: ``nurikabe_{accuracy, exact_accuracy, cell_accuracy,
    clue_accuracy, well_formed}`` (means over examples) and ``nurikabe_count``.
    ``accuracy`` is the verifier's verdict on the shading.
    """

    puzzle: kd.kontext.Key = kd.kontext.REQUIRED

    def __metric_names__(self) -> list[str]:
        return [f"nurikabe_{name}" for name in _SCORING_NAMES] + ["nurikabe_count"]

    @flax.struct.dataclass
    class State(kd.metrics.AutoState["NurikabeAllMetrics"]):
        """Collects raw tokens across batches for host-side scoring."""

        tokens: jnp.ndarray = kd.metrics.concat_field()
        ground_truth: jnp.ndarray = kd.metrics.concat_field()
        puzzle: jnp.ndarray = kd.metrics.concat_field()

        def compute(self) -> dict[str, Any]:
            """Detokenize and score all accumulated examples on host."""
            final = self.finalize()
            generated_texts = self.parent.decode_batch(np.asarray(final.tokens))
            gt_strs = self.parent.decode_batch(np.asarray(final.ground_truth))
            puzzle_strs = self.parent.decode_batch(np.asarray(final.puzzle))

            num_examples = len(generated_texts)
            totals = np.zeros(len(_ALL_SCORING_FNS), dtype=np.float64)
            for gen, gt, puz in zip(generated_texts, gt_strs, puzzle_strs):
                totals += [fn(gen, gt, puz) for fn in _ALL_SCORING_FNS]

            # Metric outputs are all routed to ``write_scalars``, so only
            # scalars belong here; board images come from ``NurikabeBoardImages``.
            result: dict[str, float] = {
                f"nurikabe_{name}": total / num_examples if num_examples else 0.0
                for name, total in zip(_SCORING_NAMES, totals)
            }
            result["nurikabe_count"] = float(num_examples)
            return result

    def get_state(
        self,
        *,
        tokens: jnp.ndarray,
        ground_truth: jnp.ndarray,
        puzzle: jnp.ndarray,
    ) -> "NurikabeAllMetrics.State":
        """Store raw tokens — all scoring is deferred to ``compute()``."""
        return self.State(
            tokens=self._squeeze_3d(tokens),
            ground_truth=self._squeeze_3d(ground_truth),
            puzzle=self._squeeze_3d(puzzle),
        )


@dataclasses.dataclass(kw_only=True, frozen=True)
class NurikabeBoardImages(base_metric.BaseTextMetric):
    """Summary: render the first ``num_images`` puzzles next to the model's board.

    Use under ``summaries=`` (not ``metrics=``): Kauldron only sends summary
    values to ``write_images``; metric values all go to ``write_scalars``.
    """

    puzzle: kd.kontext.Key = kd.kontext.REQUIRED
    num_images: int = 4

    @flax.struct.dataclass
    class State(kd.metrics.AutoState["NurikabeBoardImages"]):
        """Keeps the first ``num_images`` examples."""

        tokens: jnp.ndarray = kd.metrics.truncate_field(num_field="parent.num_images")
        ground_truth: jnp.ndarray = kd.metrics.truncate_field(
            num_field="parent.num_images"
        )
        puzzle: jnp.ndarray = kd.metrics.truncate_field(num_field="parent.num_images")

        def compute(self) -> np.ndarray:
            """Returns float32 images in [0, 1] of shape (n, h, w, 3)."""
            final = self.finalize()
            generated_texts = self.parent.decode_batch(np.asarray(final.tokens))
            gt_strs = self.parent.decode_batch(np.asarray(final.ground_truth))
            puzzle_strs = self.parent.decode_batch(np.asarray(final.puzzle))

            images = [
                nurikabe_viz.render_nurikabe(puz, gen, target=gt)
                for gen, gt, puz in zip(generated_texts, gt_strs, puzzle_strs)
            ]
            # Kauldron only writes images whose dtype is float.
            return np.stack(images).astype(np.float32) / 255.0

    def get_state(
        self,
        *,
        tokens: jnp.ndarray,
        ground_truth: jnp.ndarray,
        puzzle: jnp.ndarray,
    ) -> "NurikabeBoardImages.State":
        return self.State(
            tokens=self._squeeze_3d(tokens),
            ground_truth=self._squeeze_3d(ground_truth),
            puzzle=self._squeeze_3d(puzzle),
        )
