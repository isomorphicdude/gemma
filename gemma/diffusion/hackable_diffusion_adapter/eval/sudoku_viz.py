"""Render a Sudoku puzzle and the model's solution side by side for eval logging.

Usage:
    img = render_sudoku(puzzle_str, model_output_str)   # (H, W, 3) uint8 array
    wandb.log({"eval/sudoku": wandb.Image(img, caption=...)})
    # or TensorBoard: writer.add_image("eval/sudoku", img, step, dataformats="HWC")

Left panel: the unsolved puzzle. Right panel: the model's board, styled as below.
Pass show_puzzle=False to get only the model's board.

Cell styling (right panel):
    given clue                  -> black digit on grey
    model-filled, no conflict   -> blue digit on white
    row/col/box conflict        -> red background
    clue overwritten by model   -> orange background, shows model digit, clue in corner
    left blank (0) by model     -> hatched
    wrong vs ground truth       -> red digit, true digit in corner (only if `target` given)
"""
from __future__ import annotations

import re

import matplotlib

matplotlib.use("Agg")  # headless: safe inside eval jobs
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle

GIVEN_BG = "#e6e6e6"
GIVEN_FG = "#111111"
FILLED_FG = "#1f5fbf"
CONFLICT_BG = "#f8c4c4"
CHANGED_BG = "#ffd59e"
WRONG_FG = "#c62828"
HINT_FG = "#666666"


def parse_grid(s: str | list | np.ndarray) -> tuple[np.ndarray, bool]:
    """Parse a board into a 9x9 int array. Returns (grid, well_formed).

    Accepts "0 6 7 2 ...", "067205914...", lists, or arrays. Any non-digit text the
    model emits is ignored; short outputs are padded with 0, long ones truncated, and
    well_formed is False in either case.
    """
    if isinstance(s, (list, tuple, np.ndarray)):
        digits = [int(x) for x in np.asarray(s).ravel()]
    else:
        digits = [int(d) for d in re.findall(r"\d", s)]
    well_formed = len(digits) == 81
    digits = (digits + [0] * 81)[:81]
    return np.array(digits, dtype=int).reshape(9, 9), well_formed


def check_solution(puzzle: np.ndarray, sol: np.ndarray) -> dict:
    """Per-cell masks and summary flags for a candidate solution."""
    conflict = np.zeros((9, 9), bool)
    # rows and columns
    units = [[(r, c) for c in range(9)] for r in range(9)]
    units += [[(r, c) for r in range(9)] for c in range(9)]
    # 3x3 boxes
    units += [
        [(br + i, bc + j) for i in range(3) for j in range(3)]
        for br in (0, 3, 6)
        for bc in (0, 3, 6)
    ]
    for unit in units:
        vals = [sol[r, c] for r, c in unit]
        for (r, c), v in zip(unit, vals):
            if v != 0 and vals.count(v) > 1:
                conflict[r, c] = True

    given = puzzle != 0
    changed = given & (sol != puzzle)
    empty = sol == 0
    out_of_range = (sol < 0) | (sol > 9)
    solved = not (conflict.any() or changed.any() or empty.any() or out_of_range.any())
    return {
        "given": given,
        "conflict": conflict,
        "changed": changed,
        "empty": empty,
        "solved": solved,
        "n_conflicts": int(conflict.sum()),
        "n_changed_clues": int(changed.sum()),
        "n_empty": int(empty.sum()),
    }


def _draw_grid_lines(ax) -> None:
    for i in range(10):
        lw = 2.2 if i % 3 == 0 else 0.6
        ax.plot([0, 9], [i, i], color="black", lw=lw, zorder=2,
                solid_capstyle="projecting")
        ax.plot([i, i], [0, 9], color="black", lw=lw, zorder=2,
                solid_capstyle="projecting")


def _setup_board_ax(ax) -> None:
    ax.set_xlim(0, 9)
    ax.set_ylim(9, 0)  # row 0 at the top
    ax.set_aspect("equal")
    ax.axis("off")


def _draw_puzzle(ax, puz: np.ndarray, size_in: float) -> None:
    """Draw the unsolved puzzle: clues on grey, blanks on white."""
    _setup_board_ax(ax)
    for r in range(9):
        for c in range(9):
            v = puz[r, c]
            ax.add_patch(
                Rectangle((c, r), 1, 1, facecolor=GIVEN_BG if v else "white",
                          edgecolor="none", zorder=0)
            )
            if v:
                ax.text(c + 0.5, r + 0.54, str(v), ha="center", va="center",
                        fontsize=size_in * 4.2, color=GIVEN_FG, fontweight="bold")
    _draw_grid_lines(ax)


def _draw_solution(ax, puz, sol, tgt, info, size_in: float) -> None:
    """Draw the model's board with the cell styling described in the module doc."""
    _setup_board_ax(ax)
    for r in range(9):
        for c in range(9):
            v = sol[r, c]
            bg, fg, hatch, corner = "white", FILLED_FG, None, None

            if info["given"][r, c]:
                bg, fg = GIVEN_BG, GIVEN_FG
            if info["conflict"][r, c]:
                bg = CONFLICT_BG
            if info["changed"][r, c]:
                bg, fg, corner = CHANGED_BG, WRONG_FG, str(puz[r, c])
            if info["empty"][r, c]:
                hatch = "///"
            if tgt is not None and not info["given"][r, c] and v != tgt[r, c]:
                fg, corner = WRONG_FG, str(tgt[r, c])

            ax.add_patch(
                Rectangle((c, r), 1, 1, facecolor=bg, edgecolor="none",
                          hatch=hatch, zorder=0)
            )
            if v != 0:
                ax.text(
                    c + 0.5, r + 0.54, str(v), ha="center", va="center",
                    fontsize=size_in * 4.2, color=fg,
                    fontweight="bold" if info["given"][r, c] else "normal",
                )
            if corner is not None:
                ax.text(c + 0.88, r + 0.2, corner, ha="right", va="center",
                        fontsize=size_in * 1.8, color=HINT_FG,
                        bbox=dict(boxstyle="square,pad=0.08", fc="white",
                                  ec="none"))
    _draw_grid_lines(ax)


def _draw_no_answer(ax) -> None:
    """Draw a hatched, empty board for outputs with no digits at all."""
    _setup_board_ax(ax)
    ax.add_patch(Rectangle((0, 0), 9, 9, facecolor="white", edgecolor="none",
                           hatch="///", zorder=0))
    _draw_grid_lines(ax)


def render_sudoku(
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
    a string with no digits) is drawn as "no answer" rather than as a blank board
    that overwrote every clue.
    """
    no_answer = solution is None or (
        isinstance(solution, str) and not re.search(r"\d", solution))
    puz, _ = parse_grid(puzzle)
    sol, well_formed = parse_grid("" if solution is None else solution)
    tgt = parse_grid(target)[0] if target is not None else None
    info = check_solution(puz, sol)

    def n(k, word):
        return f"{k} {word}{'' if k == 1 else 's'}"

    status = "SOLVED" if info["solved"] else "INVALID"
    details = []
    if tgt is not None:
        n_wrong = int(((sol != tgt) & ~info["given"]).sum())
        if info["solved"] and n_wrong == 0:
            status = "SOLVED (matches target)"
        elif n_wrong:
            details.append(n(n_wrong, "cell") + " ≠ target")
    if info["n_conflicts"]:
        details.append(n(info["n_conflicts"], "conflict"))
    if info["n_changed_clues"]:
        details.append(n(info["n_changed_clues"], "clue") + " overwritten")
    if info["n_empty"]:
        details.append(f"{info['n_empty']} blank")
    if not well_formed:
        details.append("output not 81 digits")
    lines = [status]
    if details:
        lines.append(", ".join(details))
    if no_answer:
        lines = ["NO ANSWER", "no solution found in model output"]
    status_color = "#1b7f3b" if info["solved"] else WRONG_FG

    n_panels = 2 if show_puzzle else 1
    # Reserve room for two status lines even if only one is used, so every image
    # has the same size and a batch of them can be stacked.
    n_header = 3 + (1 if title else 0)
    header_h = 0.2 + 0.16 * n_header * size_in / 4  # room for the title lines
    fig, axes = plt.subplots(1, n_panels, figsize=(size_in * n_panels, size_in + header_h),
                             dpi=dpi, squeeze=False)
    axes = axes[0]

    if show_puzzle:
        _draw_puzzle(axes[0], puz, size_in)
        # Pad with blank lines so both panel titles sit on the same baseline.
        axes[0].set_title("\n" * (len(lines) - 1) + f"Puzzle ({int(info['given'].sum())} clues)",
                          fontsize=size_in * 2.3, color=GIVEN_FG)
    sol_ax = axes[-1]
    if no_answer:
        _draw_no_answer(sol_ax)
    else:
        _draw_solution(sol_ax, puz, sol, tgt, info, size_in)
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

    puzzle = ("0 6 7 2 0 5 9 1 4 3 4 0 1 0 0 0 2 0 2 0 1 6 0 0 0 3 8 0 0 0 0 4 2 0 6 0 "
              "4 0 0 7 5 1 0 0 2 9 1 2 8 0 3 7 4 0 5 0 0 4 0 6 0 8 1 0 8 0 3 1 0 0 0 9 "
              "1 2 0 0 9 0 0 7 6")
    model = ("8 6 7 2 3 5 9 1 4 3 4 5 1 8 9 6 2 7 2 9 1 6 7 4 5 3 8 7 5 8 9 4 2 1 6 3 "
             "4 3 6 7 5 1 8 9 2 9 1 2 8 6 3 7 4 5 5 7 9 4 2 6 3 8 1 6 8 4 3 1 7 2 5 9 "
             "1 2 3 5 9 8 4 7 6")

    p, _ = parse_grid(puzzle)
    s, _ = parse_grid(model)
    print(check_solution(p, s)["solved"], {k: v for k, v in check_solution(p, s).items()
                                          if k.startswith("n_")})
    Image.fromarray(render_sudoku(puzzle, model, title="example")).save("sudoku_example.png")
