# utils/radar_mty_plot.py
from __future__ import annotations
from typing import Sequence, Tuple, Optional, List
import matplotlib.pyplot as plt
from mplsoccer import Radar

# === Paleta Monterrey (ajustá si querés) ===
MTY_BLUE  = "#0B1F38"
MTY_GOLD  = "#c49308"
MTY_PALE  = "#edf2f4"
MTY_RING  = "#f2d9a6"
TXT_DARK  = "#0B1F38"
TXT_RANGE = "#6C969D"

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

def _make_radar(metrics: Sequence[str], low: Sequence[float], high: Sequence[float]) -> Radar:
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
    radar.draw_circles(ax=ax, facecolor=MTY_PALE, edgecolor=MTY_RING, lw=1.2)
    radar.draw_range_labels(ax=ax, fontsize=7, fontproperties="monospace", color=TXT_RANGE)
    for s in ax.spines.values():
        s.set_visible(False)

def _draw_param_labels(radar: Radar, ax, labels: Sequence[str]):
    radar.params = wrap_labels(labels, max_len=18)
    radar.draw_param_labels(
        ax=ax,
        fontsize=7,
        fontproperties="monospace",
        color=TXT_DARK,
        offset=1.10
    )

def _headers(ax, left: str = "", right: str = ""):
    if left:
        ax.text(
            0.0001, 1.05, left,
            fontsize=8, ha='left', va='center_baseline',
            transform=ax.transAxes, fontfamily='monospace', color=MTY_BLUE
        )
    if right:
        ax.text(
            1.0, 1.05, right,
            fontsize=8, ha='right', va='center_baseline',
            transform=ax.transAxes, fontfamily='monospace', color=MTY_GOLD
        )

def plot_radar(
    *,
    metrics: Sequence[str],
    values: Sequence[float],
    low: Sequence[float],
    high: Sequence[float],
    reference: Optional[Sequence[float]] = None,  # si pasás referencia => compare
    head_left: str = "",
    head_right: str = "",
    figsize: Tuple[float, float] = (4.2, 4.2),
):
    radar = _make_radar(metrics, low, high)
    fig, ax = plt.subplots(figsize=figsize)
    radar.setup_axis(ax=ax)
    _apply_style(ax, radar)

    if reference is None:
        radar.draw_radar_solid(
            values=values,
            ax=ax,
            kwargs={
                "facecolor": MTY_BLUE,
                "alpha": 0.60,
                "edgecolor": "none",
                "linewidth": 0,
            },
        )
    else:
        radar.draw_radar_compare(
            ax=ax,
            values=values,
            compare_values=reference,
            kwargs_radar={"facecolor": MTY_BLUE, "alpha": 0.60, "edgecolor": "none", "linewidth": 0},
            kwargs_compare={"facecolor": MTY_GOLD, "alpha": 0.55, "edgecolor": "none", "linewidth": 0},
        )

    _draw_param_labels(radar, ax, metrics)
    _headers(ax, left=head_left, right=head_right)
    fig.tight_layout(pad=0.3)
    #fig.patch.set_alpha(0)   # fondo transparente
    ax.set_facecolor("none")
    return fig
