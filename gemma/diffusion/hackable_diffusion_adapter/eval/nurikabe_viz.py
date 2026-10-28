"""Render a Nurikabe puzzle and the model's shading side by side for eval logging.

Usage:
    img = render_nurikabe(puzzle_str, model_output_str, target=solution_str)
    # (H, W, 3) uint8 array
    wandb.log({"eval/nurikabe": wandb.Image(img, caption=...)})
    # or TensorBoard: writer.add_image("eval/nurikabe", img, step, dataformats="HWC")

Boards use the dataset format: 100 space-separated cells in row-major order,
"?" for an unknown puzzle cell, "#" for sea (black) and the island size (1-10)
for an island cell (white). Validity is decided by ``nurikabe_verify.verify``.

Left panel: the puzzle. Right panel: the model's board, styled as below.
Pass show_puzzle=False to get only the model's board.

Cell styling (right panel):
    clue kept                    -> bold black number on grey
    island cell                  -> model's number in blue on white
    sea (#)                      -> dark grey
    sea inside a 2x2 block       -> dark red
    sea cut off from main sea    -> purple
    island breaking a rule       -> red background (no clue, >1 clue, wrong size)
    clue shaded or overwritten   -> orange background, clue in corner
    unknown ("?") cell           -> hatched
    shading wrong vs target      -> red corner triangle (only if `target` given)
"""
from __future__ import annotations

import re
import textwrap

import matplotlib

matplotlib.use("Agg")  # headless: safe inside eval jobs
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Polygon, Rectangle
from scipy import ndimage

from gemma.diffusion.hackable_diffusion_adapter.eval import nurikabe_verify

SIZE = 10
NUM_CELLS = SIZE * SIZE
CELL_RE = re.compile(r"#|\?|\d+")
ISLAND_SIZES = frozenset(str(k) for k in range(1, NUM_CELLS + 1))

GIVEN_BG = "#e6e6e6"
GIVEN_FG = "#111111"
SEA_BG = "#202124"
POOL_BG = "#8b1a1a"
STRAY_SEA_BG = "#5b2c83"
FILLED_FG = "#1f5fbf"
BAD_ISLAND_BG = "#f8c4c4"
CHANGED_BG = "#ffd59e"
WRONG_FG = "#c62828"
HINT_FG = "#666666"


def parse_cells(s: str | list | tuple | np.ndarray) -> tuple[list[str], bool]:
    """Split a board into 100 cell tokens. Returns (cells, well_formed).

    Accepts "1 ? # 10 ..." strings (other text is ignored) or sequences of tokens.
    Short outputs are padded with "?", long ones truncated, and well_formed is
    False in either case or when a cell is neither "#" nor an island size 1-100.
    """
    if isinstance(s, (list, tuple, np.ndarray)):
        cells = [str(x) for x in np.asarray(s).ravel()]
    else:
        cells = CELL_RE.findall(s)
    well_formed = len(cells) == NUM_CELLS and all(
        c == "#" or c in ISLAND_SIZES for c in cells)
    cells = (cells + ["?"] * NUM_CELLS)[:NUM_CELLS]
    return cells, well_formed


def clue_grid(puzzle_cells: list[str]) -> np.ndarray:
    """10x10 int clue grid: 0 for "?" (or any non-number), else the clue."""
    return np.array([int(c) if c.isdigit() else 0 for c in puzzle_cells],
                    dtype=int).reshape(SIZE, SIZE)


def shading_grid(cells: list[str]) -> np.ndarray:
    """10x10 bool grid, True for sea ("#"); every other cell counts as island."""
    return np.array([c == "#" for c in cells]).reshape(SIZE, SIZE)


def check_board(clues: np.ndarray, cells: list[str]) -> dict:
    """Per-cell masks and summary flags for a candidate board."""
    black = shading_grid(cells)
    given = clues > 0
    tokens = np.array(cells, dtype=object).reshape(SIZE, SIZE)
    errors = nurikabe_verify.verify(clues.tolist(), black.astype(int).tolist())

    # 2x2 all-sea blocks: mark all four cells of each one.
    block = black[:-1, :-1] & black[:-1, 1:] & black[1:, :-1] & black[1:, 1:]
    pool = np.zeros_like(black)
    for dr in (0, 1):
        for dc in (0, 1):
            pool[dr:SIZE - 1 + dr, dc:SIZE - 1 + dc] |= block

    # Sea outside its largest 4-connected region (ties keep the first).
    sea_labels, n_sea = ndimage.label(black)
    stray_sea = np.zeros_like(black)
    if n_sea > 1:
        main = 1 + int(np.argmax(np.bincount(sea_labels.ravel())[1:]))
        stray_sea = black & (sea_labels != main)

    # Islands that do not hold exactly one clue equal to their size.
    island_labels, n_islands = ndimage.label(~black)
    bad_island = np.zeros_like(black)
    n_bad_islands = 0
    for k in range(1, n_islands + 1):
        region = island_labels == k
        region_clues = clues[region & given]
        if len(region_clues) != 1 or region.sum() != region_clues[0]:
            bad_island |= region
            n_bad_islands += 1

    changed = given & (tokens != clues.astype(str))
    unknown = np.array([c != "#" and c not in ISLAND_SIZES for c in cells]).reshape(
        SIZE, SIZE)
    return {
        "black": black,
        "given": given,
        "pool": pool,
        "stray_sea": stray_sea,
        "bad_island": bad_island,
        "changed": changed,
        "unknown": unknown,
        "errors": errors,
        "valid": not errors and not unknown.any(),
        "n_pools": int(block.sum()),
        "n_sea_regions": int(n_sea),
        "n_bad_islands": n_bad_islands,
        "n_changed_clues": int(changed.sum()),
        "n_unknown": int(unknown.sum()),
    }


def _draw_grid_lines(ax) -> None:
    for i in range(SIZE + 1):
        lw = 2.2 if i in (0, SIZE) else 0.6
        ax.plot([0, SIZE], [i, i], color="black", lw=lw, zorder=2,
                solid_capstyle="projecting")
        ax.plot([i, i], [0, SIZE], color="black", lw=lw, zorder=2,
                solid_capstyle="projecting")


def _setup_board_ax(ax) -> None:
    ax.set_xlim(0, SIZE)
    ax.set_ylim(SIZE, 0)  # row 0 at the top
    ax.set_aspect("equal")
    ax.axis("off")


def _draw_puzzle(ax, clues: np.ndarray, size_in: float) -> None:
    """Draw the unsolved puzzle: clues on grey, unknown cells on white."""
    _setup_board_ax(ax)
    for r in range(SIZE):
        for c in range(SIZE):
            v = clues[r, c]
            ax.add_patch(
                Rectangle((c, r), 1, 1, facecolor=GIVEN_BG if v else "white",
                          edgecolor="none", zorder=0)
            )
            if v:
                ax.text(c + 0.5, r + 0.54, str(v), ha="center", va="center",
                        fontsize=size_in * 3.4, color=GIVEN_FG, fontweight="bold")
    _draw_grid_lines(ax)


def _draw_solution(ax, clues, cells, tgt_black, info, size_in: float) -> None:
    """Draw the model's board with the cell styling described in the module doc."""
    _setup_board_ax(ax)
    for r in range(SIZE):
        for c in range(SIZE):
            token = cells[r * SIZE + c]
            bg, fg, hatch, corner = "white", FILLED_FG, None, None
            given = info["given"][r, c]

            if info["black"][r, c]:
                bg = SEA_BG
                if info["stray_sea"][r, c]:
                    bg = STRAY_SEA_BG
                if info["pool"][r, c]:
                    bg = POOL_BG
            else:
                if given:
                    bg, fg = GIVEN_BG, GIVEN_FG
                if info["bad_island"][r, c]:
                    bg = BAD_ISLAND_BG
            if info["changed"][r, c]:
                bg, fg, corner = CHANGED_BG, WRONG_FG, str(clues[r, c])
            if info["unknown"][r, c]:
                hatch = "///"

            ax.add_patch(
                Rectangle((c, r), 1, 1, facecolor=bg, edgecolor="none",
                          hatch=hatch, zorder=0)
            )
            if token != "#" and token != "?":
                ax.text(
                    c + 0.5, r + 0.54, token, ha="center", va="center",
                    fontsize=size_in * 3.4, color=fg,
                    fontweight="bold" if given else "normal",
                )
            if corner is not None:
                ax.text(c + 0.9, r + 0.2, corner, ha="right", va="center",
                        fontsize=size_in * 1.6, color=HINT_FG,
                        bbox=dict(boxstyle="square,pad=0.08", fc="white",
                                  ec="none"))
            if tgt_black is not None and info["black"][r, c] != tgt_black[r, c]:
                ax.add_patch(Polygon(
                    [(c, r), (c + 0.35, r), (c, r + 0.35)], closed=True,
                    facecolor=WRONG_FG, edgecolor="white", lw=0.8, zorder=1))
    _draw_grid_lines(ax)


def _draw_no_answer(ax) -> None:
    """Draw a hatched, empty board for outputs with no cells at all."""
    _setup_board_ax(ax)
    ax.add_patch(Rectangle((0, 0), SIZE, SIZE, facecolor="white", edgecolor="none",
                           hatch="///", zorder=0))
    _draw_grid_lines(ax)


def render_nurikabe(
    puzzle,
    solution,
    target=None,
    title: str | None = None,
    size_in: float = 4.0,
    dpi: int = 100,
    show_puzzle: bool = True,
) -> np.ndarray:
    """Draw the puzzle and the model's board side by side; return (H, W, 3) uint8.

    With show_puzzle=False only the model's board is drawn. ``solution=None`` (or
    a string with no cells) is drawn as "no answer". Every call with the same
    ``size_in``, ``dpi``, ``show_puzzle`` and ``title`` presence returns the same
    image size, so a batch of them can be stacked.
    """
    no_answer = solution is None or (
        isinstance(solution, str) and not CELL_RE.search(solution))
    clues = clue_grid(parse_cells(puzzle)[0])
    cells, well_formed = parse_cells("" if solution is None else solution)
    tgt_black = shading_grid(parse_cells(target)[0]) if target is not None else None
    info = check_board(clues, cells)

    def n(k, word):
        return f"{k} {word}{'' if k == 1 else 's'}"

    solved = info["valid"] and well_formed
    status = "VALID" if solved else "INVALID"
    details = []
    if tgt_black is not None:
        n_wrong = int((info["black"] != tgt_black).sum())
        if solved and n_wrong == 0:
            status = "VALID (matches target)"
        elif n_wrong:
            details.append(n(n_wrong, "cell") + " ≠ target")
    if info["n_pools"]:
        details.append(n(info["n_pools"], "2x2 pool"))
    if info["n_sea_regions"] > 1:
        details.append(f"sea in {info['n_sea_regions']} parts")
    if info["n_bad_islands"]:
        details.append(n(info["n_bad_islands"], "bad island"))
    if info["n_changed_clues"]:
        details.append(n(info["n_changed_clues"], "clue") + " changed")
    if info["n_unknown"]:
        details.append(f"{info['n_unknown']} unknown")
    if not well_formed:
        details.append("output not 100 cells")
    # At most two detail lines, so the header height is fixed.
    lines = [status] + textwrap.wrap(", ".join(details), width=44)[:2]
    if no_answer:
        lines = ["NO ANSWER", "no cells found in model output"]
    status_color = "#1b7f3b" if solved else WRONG_FG

    n_panels = 2 if show_puzzle else 1
    # Reserve room for three status lines even if fewer are used, so every image
    # has the same size and a batch of them can be stacked.
    n_header = 4 + (1 if title else 0)
    header_h = 0.2 + 0.16 * n_header * size_in / 4
    fig, axes = plt.subplots(1, n_panels, figsize=(size_in * n_panels, size_in + header_h),
                             dpi=dpi, squeeze=False)
    axes = axes[0]

    if show_puzzle:
        _draw_puzzle(axes[0], clues, size_in)
        # Pad with blank lines so both panel titles sit on the same baseline.
        axes[0].set_title("\n" * (len(lines) - 1) + f"Puzzle ({int(info['given'].sum())} clues)",
                          fontsize=size_in * 2.3, color=GIVEN_FG)
    sol_ax = axes[-1]
    if no_answer:
        _draw_no_answer(sol_ax)
    else:
        _draw_solution(sol_ax, clues, cells, tgt_black, info, size_in)
    sol_ax.set_title("\n".join(lines), fontsize=size_in * 2.3, color=status_color)

    if title:
        fig.suptitle(title, fontsize=size_in * 2.6)

    fig.tight_layout(pad=0.3, w_pad=1.0)
    fig.canvas.draw()
    img = np.asarray(fig.canvas.buffer_rgba())[..., :3].copy()
    plt.close(fig)  # important in an eval loop, otherwise figures pile up in memory
    return img


if __name__ == "__main__":
    from PIL import Image

    puzzle = ("1 ? 1 ? 1 ? 1 ? ? ? ? ? ? ? ? ? ? ? 1 ? ? 1 ? 2 ? 1 ? 1 ? ? ? ? ? ? ? ? "
              "? ? 1 ? ? 1 ? ? ? 1 ? 1 ? ? ? ? 1 ? 1 ? ? ? 1 ? ? 1 ? ? ? ? 1 ? ? ? ? ? "
              "? 1 ? ? ? ? 1 ? ? 1 ? ? 6 ? ? ? ? ? ? ? ? ? ? ? ? ? 1 ?")
    target = ("1 # 1 # 1 # 1 # # # # # # # # # # # 1 # # 1 # 2 # 1 # 1 # # # # # 2 # # "
              "# # 1 # # 1 # # # 1 # 1 # # # # 1 # 1 # # # 1 # # 1 # # # # 1 # # # # # "
              "# 1 # 6 # # 1 # # 1 # # 6 6 6 # # # # # # 6 6 # # # 1 #")
    # Break the target: shade the 2's partner cell, making a pool and a short island.
    model = target.split()
    model[33] = "#"
    model = " ".join(model)

    for name, board in (("target", target), ("model", model)):
        info = check_board(clue_grid(parse_cells(puzzle)[0]), parse_cells(board)[0])
        print(name, info["valid"], info["errors"])
    Image.fromarray(render_nurikabe(puzzle, model, target=target, title="example")).save(
        "nurikabe_example.png")
