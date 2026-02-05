from __future__ import annotations
from typing import Sequence, List
import matplotlib.pyplot as plt
from mplsoccer import Radar
from matplotlib import colors as mcolors

# === Paleta Monterrey ===
MTY_BLUE  = "#0B1F38"
MTY_GOLD  = "#c49308"
MTY_RING  = "#FFFFFF"
ACCENT = "#6CA0DC" 
TXT_LIGHT = "#FFFFFF"
TXT_RANGE = "#C7D2E3"

# -----------------------
# Helpers
# -----------------------
def wrap_labels(labels: Sequence[str], max_len: int = 18) -> List[str]:
    out = []
    for label in labels:
        words = str(label).split()
        line, lines = "", []
        for w in words:
            if len((line + " " + w).strip()) <= max_len:
                line = (line + " " + w).strip()
            else:
                if line:
                    lines.append(line)
                line = w
        if line:
            lines.append(line)
        out.append("\n".join(lines))
    return out


def _make_radar(metrics, low, high) -> Radar:
    return Radar(
        params=list(metrics),
        min_range=list(low),
        max_range=list(high),
        round_int=[False] * len(metrics),
        num_rings=4,
        ring_width=1,
        center_circle_radius=1,
    )


def _apply_style(ax, radar: Radar):
    ring_rgba = mcolors.to_rgba(MTY_RING, 0.14)

    # ⚠️ IMPORTANTE:
    # NO pintamos fondo → transparencia real
    radar.draw_circles(
        ax=ax,
        facecolor="none",
        edgecolor=ring_rgba,
        lw=1.1
    )

    radar.draw_range_labels(
        ax=ax,
        fontsize=7,
        fontproperties="monospace",
        color=TXT_RANGE
    )

    for s in ax.spines.values():
        s.set_visible(False)


def _draw_param_labels(radar: Radar, ax, labels):
    radar.params = wrap_labels(labels, max_len=18)
    radar.draw_param_labels(
        ax=ax,
        fontsize=7,
        fontproperties="monospace",
        color=TXT_LIGHT,
        offset=1.10
    )


def _headers(ax, left: str = "", right: str = ""):
    if left:
        ax.text(
            0.0001, 1.05, left,
            fontsize=8,
            ha="left",
            va="center_baseline",
            transform=ax.transAxes,
            fontfamily="monospace",
            color=TXT_LIGHT,
        )
    if right:
        ax.text(
            1.0, 1.05, right,
            fontsize=8,
            ha="right",
            va="center_baseline",
            transform=ax.transAxes,
            fontfamily="monospace",
            color=MTY_GOLD,
        )


# -----------------------
# Main
# -----------------------
def plot_radar(
    *,
    metrics,
    values,
    low,
    high,
    reference=None,
    head_left="",
    head_right="",
    figsize=(4.2, 4.2),
):
    radar = _make_radar(metrics, low, high)

    fig, ax = plt.subplots(figsize=figsize)

    # ✅ transparencia real
    fig.patch.set_alpha(0)
    ax.set_facecolor("none")

    radar.setup_axis(ax=ax)
    _apply_style(ax, radar)

    if reference is None:
        radar.draw_radar_solid(
            values=values,
            ax=ax,
            kwargs={
                "facecolor": ACCENT,
                "alpha": 0.55,
                "edgecolor": "none",
                "linewidth": 0,
            },
        )
    else:
        radar.draw_radar_compare(
            ax=ax,
            values=values,
            compare_values=reference,
            kwargs_radar={
                "facecolor": ACCENT,
                "alpha": 0.55,
                "edgecolor": "none",
                "linewidth": 0,
            },
            kwargs_compare={
                "facecolor": MTY_GOLD,
                "alpha": 0.45,
                "edgecolor": "none",
                "linewidth": 0,
            },
        )

    _draw_param_labels(radar, ax, metrics)
    _headers(ax, left=head_left, right=head_right)

    fig.tight_layout(pad=0.3)
    return fig
