from __future__ import annotations
from typing import Sequence, List
import matplotlib.pyplot as plt
from mplsoccer import Radar
from matplotlib import colors as mcolors
from utils.text_wrapper import wrap_two_lines


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


def _split_head_left(head_left: str) -> tuple[str, str]:
    """
    Espera algo tipo: "Jugador | Equipo"
    Si no hay '|', devuelve todo como jugador y team vacío.
    """
    s = (head_left or "").strip()
    if "|" not in s:
        return s, ""
    parts = [p.strip() for p in s.split("|")]
    player = parts[0] if parts else s
    team = " | ".join(parts[1:]).strip()
    return player, team


def _headers(ax, left: str = "", right: str = ""):
    # --- LEFT: Jugador arriba, Equipo abajo (y dinámico según wrap)
    if left:
        player, team = _split_head_left(left)

        # wrap a máx 2 líneas
        player_wrapped = wrap_two_lines(player, max_chars=30)
        team_wrapped   = wrap_two_lines(team,   max_chars=34)

        # cuántas líneas ocupa el jugador
        player_lines = player_wrapped.count("\n") + 1

        # posición dinámica del equipo:
        # si el jugador ocupa 2 líneas, bajamos el equipo un poco más
        y_player = 1.075
        y_team = 1.015 - (player_lines - 1) * 0.030

        # jugador
        ax.text(
            0.0001, y_player, player_wrapped,
            fontsize=8.6,
            ha="left",
            va="center_baseline",
            transform=ax.transAxes,
            fontfamily="monospace",
            color=TXT_LIGHT,
            clip_on=False,
        )

        # equipo
        if team_wrapped:
            ax.text(
                0.0001, y_team, team_wrapped,
                fontsize=7.4,
                ha="left",
                va="center_baseline",
                transform=ax.transAxes,
                fontfamily="monospace",
                color=TXT_RANGE,
                clip_on=False,
            )

    # --- RIGHT: Promedio / Otro jugador (wrap fijo)
    if right:
        right_wrapped = wrap_two_lines(right, max_chars=24)

        ax.text(
            1.0, 1.075, right_wrapped,
            fontsize=8.6,
            ha="right",
            va="center_baseline",
            transform=ax.transAxes,
            fontfamily="monospace",
            color=MTY_GOLD,
            clip_on=False,
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

    fig.tight_layout(pad=0.55)
    return fig
