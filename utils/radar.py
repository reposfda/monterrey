# utils/radar.py
from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt


def radar_plot(
    labels: list[str],
    series: list[tuple[str, list[float]]],
    *,
    title: str = "",
    vmin: float = 0,
    vmax: float = 100,
):
    """
    labels: ejes
    series: lista de (nombre, valores)
    valores en [vmin, vmax] idealmente.
    """
    N = len(labels)
    if N < 3:
        raise ValueError("Radar requiere al menos 3 ejes.")

    angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
    angles += angles[:1]

    fig = plt.figure(figsize=(6.6, 6.6), dpi=140)
    ax = plt.subplot(111, polar=True)
    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels, fontsize=10)

    ax.set_ylim(vmin, vmax)
    ax.set_yticks([vmin, (vmin+vmax)/2, vmax])
    ax.set_yticklabels([str(int(vmin)), str(int((vmin+vmax)/2)), str(int(vmax))], fontsize=9)

    for name, vals in series:
        vals = [float(x) if x is not None else np.nan for x in vals]
        vals = np.clip(vals, vmin, vmax).tolist()
        vals += vals[:1]
        ax.plot(angles, vals, linewidth=2, label=name)
        ax.fill(angles, vals, alpha=0.12)

    if title:
        ax.set_title(title, fontsize=12, pad=18)

    ax.legend(loc="upper right", bbox_to_anchor=(1.25, 1.15), frameon=False)
    fig.tight_layout()
    return fig
