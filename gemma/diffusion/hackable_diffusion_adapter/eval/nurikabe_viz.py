import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np

def plot_nurikabe(clues, black=None):
    """Return a figure showing the clues and, optionally, a solved shading."""
    height, width = clues.shape
    panels = [("Puzzle", None)]
    if black is not None:
        panels.append(("Saved solution", black))

    given_bg, given_fg, sea_bg = "#e6e6e6", "#111111", "#202124"
    panel_height = 4.8
    panel_width = panel_height * width / height
    fig, axes = plt.subplots(
        1, len(panels), figsize=(panel_width * len(panels), panel_height + 0.5),
        squeeze=False, facecolor="white",
    )
    for ax, (title, shading) in zip(axes[0], panels):
        ax.set_xlim(0, width)
        ax.set_ylim(height, 0)  # row 0 is at the top
        ax.set_aspect("equal")
        ax.axis("off")
        for r in range(height):
            for c in range(width):
                value = int(clues[r, c])
                background = given_bg if value else "white"
                if shading is not None and shading[r, c]:
                    background = sea_bg
                ax.add_patch(Rectangle(
                    (c, r), 1, 1, facecolor=background, edgecolor="none", zorder=0,
                ))
                if value:
                    ax.text(
                        c + 0.5, r + 0.53, str(value), ha="center", va="center",
                        fontsize=min(20, 160 / max(height, width)),
                        fontweight="bold", color=given_fg, zorder=3,
                    )
        for r in range(height + 1):
            ax.plot([0, width], [r, r], color="black",
                    lw=2.2 if r in (0, height) else 0.6, zorder=2,
                    solid_capstyle="projecting", clip_on=False)
        for c in range(width + 1):
            ax.plot([c, c], [0, height], color="black",
                    lw=2.2 if c in (0, width) else 0.6, zorder=2,
                    solid_capstyle="projecting", clip_on=False)
        ax.set_title(title, fontsize=14, color=given_fg, pad=12)
    fig.tight_layout(pad=1.0, w_pad=2.0)
    return fig
