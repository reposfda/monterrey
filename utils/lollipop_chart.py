from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from utils.text_wrapper import wrap_two_lines


# Colores default (si no los pasás por parámetro)
PRIMARY_BG = "#0B1F38"
SECONDARY_BG = "#091325"
ACCENT = "#6CA0DC"
TEXT = "#FFFFFF"
GOLD = "#c49308"
GOLD_SOFT = "#d6b35f"

def plot_lollipop_mty(
    labels: list[str],
    values: list[float],
    reference: list[float] | None,
    *,
    xlim: tuple[float, float],
    title_left: str,
    title_right: str = "",
    value_fmt: str = "{:.0f}",
    mode: str = "percentil",          # "percentil" | "real"
    reference_kind: str = "player",   # "avg" | "player"
    show_value_annotations: bool = True,
    fig_w: float = 7.0,
    row_h: float = 0.30,
    min_h: float = 3.2,
    font_family: str = "monospace",   # ✅ igualalo con radar
) -> "plt.Figure":
    """
    Lollipop horizontal estilo Monterrey:
    - Orden por categoría (parsea 'Categoria: metrica')
    - Separadores punteados entre categorías
    - Promedio en percentil: línea vertical punteada (en vez de puntos)
    - Comparar con otro jugador: segundo punto + dumbbell
    """

    # --- construir df y extraer categoría desde el prefijo antes de ":"
    cats = []
    clean_labels = []
    for lab in labels:
        s = str(lab)
        if ":" in s:
            c, rest = s.split(":", 1)
            cats.append(c.strip())
            clean_labels.append(rest.strip())
        else:
            cats.append("Otros")
            clean_labels.append(s)

    dfp = pd.DataFrame(
        {"category": cats, "label": clean_labels, "value": values, "idx": np.arange(len(values))}
    )
    if reference is not None:
        dfp["ref"] = reference

    # mantener filas con value válido (no mates la ref)
    dfp = dfp.dropna(subset=["value"]).copy()
    if dfp.empty:
        fig = plt.figure(figsize=(fig_w, min_h))
        fig.patch.set_facecolor(PRIMARY_BG)
        return fig

    # --- orden por categoría (en el orden que aparecen)
    cat_order = {c: i for i, c in enumerate(pd.unique(dfp["category"]))}
    dfp["cat_ord"] = dfp["category"].map(cat_order)

    # dentro de cada categoría: mantener orden original (idx)
    dfp = dfp.sort_values(["cat_ord", "idx"], ascending=True).reset_index(drop=True)

    # --- y con gaps entre categorías (1 fila vacía)
    gap = 1.0  # 1.0 = equivalente a una métrica; probá 0.8–1.4
    y = []
    current = 0.0
    prev_cat = None

    for _, row in dfp.iterrows():
        if prev_cat is not None and row["category"] != prev_cat:
            current += gap  # ✅ deja un espacio vacío
        y.append(current)
        current += 1.0
        prev_cat = row["category"]

    y = np.array(y, dtype="float64")

    fig_h = max(min_h, row_h * (len(dfp) + 0.8 * len(dfp["category"].unique())) + 1.1)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    fig.patch.set_facecolor(PRIMARY_BG)
    ax.set_facecolor(PRIMARY_BG)

    # grid
    ax.grid(axis="x", color="white", alpha=0.08, linewidth=1)
    ax.grid(axis="y", color="white", alpha=0.0)

    # baseline + lollipops (jugador)
    for yi, v in zip(y, dfp["value"].tolist()):
        ax.hlines(yi, xlim[0], v, color=ACCENT, alpha=0.35, linewidth=3, zorder=1)


    ax.scatter(
        dfp["value"], y,
        s=90, color=ACCENT, edgecolor="white", linewidth=1.2,
        zorder=3,
        label=wrap_two_lines("Jugador", max_chars=22),
    )


    # --- referencia
    is_percentil = (mode == "percentil")

    if reference is not None and "ref" in dfp.columns:
        if is_percentil and reference_kind == "avg":
            # ✅ promedio como línea vertical punteada (no ensucia)
            # si tu promedio es fijo 50, esto es perfecto. Si es por-métrica, igual sirve pero sería raro.
            # Tomamos la mediana/50 si vienen valores.
            x_ref = 50.0 if (xlim == (0, 100)) else float(np.nanmedian(dfp["ref"].values))
            ax.axvline(x_ref, color=GOLD, linestyle=(0, (3, 3)), linewidth=2.0, alpha=0.9, zorder=2)
            # etiqueta arriba
            if title_right:
                ax.text(
                    x_ref,
                    1.02,
                    title_right,
                    transform=ax.get_xaxis_transform(),
                    ha="center",
                    va="bottom",
                    color=GOLD,
                    fontsize=9.5,
                    fontfamily=font_family,
                    fontweight="bold",
                )
        else:
            # ✅ compare jugador (punto + dumbbell)
            ref_label = wrap_two_lines((title_right or "Referencia"), max_chars=22)

            ax.scatter(
                dfp["ref"], y,
                s=60, color=GOLD, edgecolor="white", linewidth=1.0,
                zorder=4,
                label=ref_label,  # ✅ wrap ya en el label
            )
            for yi, v, r in zip(y, dfp["value"].tolist(), dfp["ref"].tolist()):
                if pd.isna(r):
                    continue
                ax.hlines(yi, min(v, r), max(v, r), color="white", alpha=0.10, linewidth=2, zorder=2)


    # y labels (con categoría “implícita” en el texto de la izquierda si querés)
    ax.set_yticks(y)
    ax.set_yticklabels(dfp["label"].tolist(), fontsize=10.5, color=TEXT, fontfamily=font_family)
    ax.set_ylim(y.max() + 0.8, -0.8)

    ax.set_xlim(*xlim)
    ax.tick_params(axis="x", colors=TEXT, labelsize=10)
    for tick in ax.get_xticklabels():
        tick.set_fontfamily(font_family)

    for s in ax.spines.values():
        s.set_visible(False)

    # títulos
    ax.text(
        0.0, 1.04, title_left,
        transform=ax.transAxes,
        ha="left", va="bottom",
        color=TEXT,
        fontsize=13,
        fontweight="bold",
        fontfamily=font_family,
    )

    # annotations (solo si no está superpoblado)
    if show_value_annotations:
        for yi, v in zip(y, dfp["value"].tolist()):
            ax.text(
                v, yi, "  " + value_fmt.format(v),
                ha="left", va="center",
                fontsize=9.2, color="white", alpha=0.85,
                zorder=5, fontfamily=font_family,
            )

        if reference is not None and "ref" in dfp.columns and not (is_percentil and reference_kind == "avg"):
            for yi, r in zip(y, dfp["ref"].tolist()):
                if pd.isna(r):
                    continue
                ax.text(
                    r, yi, "  " + value_fmt.format(r),
                    ha="left", va="center",
                    fontsize=9.0, color=GOLD, alpha=0.95,
                    zorder=6, fontfamily=font_family,
                )

    # --- separadores en el gap + título pegado a la línea (mismo y)
    cats_sorted = dfp["category"].tolist()

    # 1) líneas punteadas y posición y_line por corte
    cut_lines = []  # (i, y_line) guardamos para ubicar texto
    for i in range(1, len(dfp)):
        if cats_sorted[i] != cats_sorted[i - 1]:
            y_line = (y[i - 1] + y[i]) / 2.0
            cut_lines.append((i, y_line))

            ax.hlines(
                y_line,
                xlim[0],
                xlim[1],
                color=GOLD_SOFT,
                alpha=0.20,
                linewidth=1.2,
                linestyle=(0, (3, 3)),
                zorder=0,
            )

    # 2) texto de categoría en el MISMO y_line (un poquito arriba para que no lo cruce la línea)
    TEXT_DY = 0.0  # subí/bajá (0.12–0.28)

    for i, y_line in cut_lines:
        # la categoría que termina arriba del corte (grupo anterior)
        cat = cats_sorted[i - 1]

        ax.text(
            -0.02,
            y_line - TEXT_DY,  # ✅ mismo y, apenas arriba de la línea (recordá eje invertido)
            cat,
            transform=ax.get_yaxis_transform(),
            ha="right",
            va="center",
            color=GOLD_SOFT,
            alpha=0.78,
            fontsize=10,
            fontfamily=font_family,
            fontweight="bold",
            zorder=10,
        )

    # 3) línea final + título de la última categoría (para que no quede suelta)
    last_cat = cats_sorted[-1]
    last_idx = len(dfp) - 1

    # y "virtual" abajo del último grupo (usa el mismo gap que venís usando)
    y_end = y[last_idx] + 0.7 * gap

    # línea punteada final (opcional, queda prolijo)
    ax.hlines(
        y_end,
        xlim[0],
        xlim[1],
        color=GOLD_SOFT,
        alpha=0.20,
        linewidth=1.2,
        linestyle=(0, (3, 3)),
        zorder=0,
    )

    # título en ese mismo y_end (como los demás)
    ax.text(
        -0.02,
        y_end - TEXT_DY,
        last_cat,
        transform=ax.get_yaxis_transform(),
        ha="right",
        va="center",
        color=GOLD_SOFT,
        alpha=0.78,
        fontsize=10,
        fontfamily=font_family,
        fontweight="bold",
        zorder=10,
    )

    # legend (afuera)
    if reference is not None and not (is_percentil and reference_kind == "avg"):
        leg = ax.legend(
            loc="upper left",
            bbox_to_anchor=(1.01, 1.00),
            borderaxespad=0.0,
            frameon=True,
            facecolor=SECONDARY_BG,
            edgecolor="none",
            fontsize=9.2,
        )

        # ✅ aplicar wrapper a los textos reales de la leyenda
        for t in leg.get_texts():
            t.set_text(wrap_two_lines(t.get_text(), max_chars=22))
            t.set_color(TEXT)
            t.set_fontfamily(font_family)

    fig.subplots_adjust(left=0.40, right=0.76, top=0.90, bottom=0.10)
    return fig
