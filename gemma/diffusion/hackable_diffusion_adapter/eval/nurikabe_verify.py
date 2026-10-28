#!/usr/bin/env python3
"""Nurikabe solution verifier.

A shading is a valid Nurikabe solution when:
  1. every clue cell is white;
  2. no 2x2 block is entirely black;
  3. all black cells form one orthogonally connected region;
  4. every white region (island) contains exactly one clue, and its number
     of cells equals that clue (so islands never touch side-by-side).

Library use:
    from nurikabe_verify import verify
    errors = verify(clues, black)   # [] means valid

    clues[r][c] is 0 (empty) or k > 0 (clue k); black[r][c] is 1 (black, wall)
    or 0 (white, island). Any 2D sequence of ints works, including numpy
    arrays; reshape flattened rows and threshold model outputs to 0/1 first.
    Messages use 0-based (row, col), and islands are named by their first cell
    in reading order. Runs in O(rows * cols).

Command line:
    python3 nurikabe_verify.py board.txt   (or pipe the board on stdin)
    Exit code 0 = valid, 1 = invalid, 2 = unreadable input.

Board file: the clue grid, a blank line, then the shading grid, e.g.

    2 . .
    . . 1

    .##
    .#.

Clue grid tokens are separated by whitespace: '.' or 0 is an empty cell and a
positive integer is a clue. Shading cells are '#' or 1 for black and '.' or 0
for white; spaces between them are optional. nurikabe_verify.cpp reads the
same format and prints the same messages.
"""

import sys
from typing import List, Sequence, Tuple

Grid = Sequence[Sequence[int]]


def verify(clues: Grid, black: Grid) -> List[str]:
    """Return every rule the shading breaks; an empty list means it is a valid solution."""
    try:
        C = [[int(v) for v in row] for row in clues]
    except (TypeError, ValueError):
        return ["clue grid must be a 2D grid of integers"]
    try:
        B = [[int(v) for v in row] for row in black]
    except (TypeError, ValueError):
        return ["shading grid must be a 2D grid of 0/1 values"]

    R = len(C)
    if R == 0 or len(C[0]) == 0:
        return ["clue grid is empty"]
    W = len(C[0])
    if any(len(row) != W for row in C):
        return ["clue grid rows have different lengths"]
    if len(B) != R or any(len(row) != W for row in B):
        return [f"shading grid must be {R}x{W} to match the clue grid"]

    errors: List[str] = []
    for r in range(R):
        for c in range(W):
            if C[r][c] < 0:
                errors.append(f"invalid clue {C[r][c]} at ({r}, {c})")
            if B[r][c] not in (0, 1):
                errors.append(f"invalid shading value {B[r][c]} at ({r}, {c})")
    if errors:
        return errors

    # Rule 1: clue cells are white.
    for r in range(R):
        for c in range(W):
            if C[r][c] and B[r][c]:
                errors.append(f"clue {C[r][c]} at ({r}, {c}) is shaded black")

    # Rule 2: no 2x2 all-black block.
    for r in range(R - 1):
        for c in range(W - 1):
            if B[r][c] and B[r][c + 1] and B[r + 1][c] and B[r + 1][c + 1]:
                errors.append(f"2x2 black block with top-left at ({r}, {c})")

    # Rules 3 and 4: flood-fill every same-colored region, scanning in reading order.
    seen = [[False] * W for _ in range(R)]
    black_regions = 0
    island_errors: List[str] = []
    for r0 in range(R):
        for c0 in range(W):
            if seen[r0][c0]:
                continue
            color = B[r0][c0]
            seen[r0][c0] = True
            stack = [(r0, c0)]
            size = 0
            clue_cells = []
            while stack:
                r, c = stack.pop()
                size += 1
                if C[r][c]:
                    clue_cells.append((r, c))
                for nr, nc in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
                    if 0 <= nr < R and 0 <= nc < W and not seen[nr][nc] and B[nr][nc] == color:
                        seen[nr][nc] = True
                        stack.append((nr, nc))
            if color == 1:
                black_regions += 1
            elif not clue_cells:
                island_errors.append(f"island at ({r0}, {c0}) has no clue (size {size})")
            elif len(clue_cells) > 1:
                island_errors.append(f"island at ({r0}, {c0}) contains {len(clue_cells)} clues")
            else:
                cr, cc = clue_cells[0]
                if size != C[cr][cc]:
                    island_errors.append(f"island with clue {C[cr][cc]} at ({cr}, {cc}) has size {size}")

    if black_regions > 1:
        errors.append(f"black cells form {black_regions} separate regions")
    return errors + island_errors


def parse(text: str) -> Tuple[List[List[int]], List[List[int]]]:
    """Parse a board file: the clue grid, a blank line, then the shading grid."""
    blocks: List[List[str]] = []
    current: List[str] = []
    for line in text.splitlines():
        if line.strip():
            current.append(line)
        elif current:
            blocks.append(current)
            current = []
    if current:
        blocks.append(current)
    if len(blocks) != 2:
        raise ValueError("expected a clue grid, a blank line, then a shading grid")

    clues = []
    for line in blocks[0]:
        row = []
        for tok in line.split():
            if tok == ".":
                row.append(0)
            elif tok.isascii() and tok.isdigit() and len(tok) <= 9:
                row.append(int(tok))
            else:
                raise ValueError(f"bad clue token {tok!r}")
        clues.append(row)

    black = []
    for line in blocks[1]:
        row = []
        for ch in line:
            if ch in "#1":
                row.append(1)
            elif ch in ".0":
                row.append(0)
            elif not ch.isspace():
                raise ValueError(f"bad shading character {ch!r}")
        black.append(row)
    return clues, black


def main(argv: List[str]) -> int:
    if len(argv) > 1 and argv[1] in ("-h", "--help"):
        print(__doc__)
        return 0
    try:
        if len(argv) > 1:
            with open(argv[1], encoding="utf-8") as f:
                text = f.read()
        else:
            text = sys.stdin.read()
        clues, black = parse(text)
    except (OSError, ValueError) as e:
        print(f"error: {e}", file=sys.stderr)
        return 2
    errors = verify(clues, black)
    if not errors:
        print("valid")
        return 0
    print("invalid")
    for e in errors:
        print(f"  - {e}")
    return 1


if __name__ == "__main__":
    sys.exit(main(sys.argv))
