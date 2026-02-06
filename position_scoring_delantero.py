# -*- coding: utf-8 -*-
"""
Sistema de Scoring para Delanteros
Versión df-first - usa df o CSV per90 (sin pool_builder, sin cálculo de métricas)

Categorías:
1. Finalización / Killer
2. Presionante
3. Conector / Falso 9
4. Disruptivo

Requiere:
- outputs/all_players_complete_2025_2026.csv (o df equivalente)
- positions_config.py
"""

from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd

from positions_config import normalize_group, sb_positions_for
from utils.scoring_io import read_input  # <-- ajustá si tu ruta difiere


# ============= HELPERS =============
def safe_numeric(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce")


def pct_rank_0_100(s: pd.Series) -> pd.Series:
    x = s.copy()
    m = x.notna()
    out = pd.Series(np.nan, index=x.index, dtype="float64")
    out.loc[m] = x.loc[m].rank(pct=True, method="average") * 100.0
    return out


def wavg(df: pd.DataFrame, cols_weights):
    cols = [c for c, _ in cols_weights if c in df.columns]
    if not cols:
        return pd.Series(np.nan, index=df.index)

    w = np.array([w for c, w in cols_weights if c in df.columns], dtype="float64")
    if w.sum() <= 0:
        return pd.Series(np.nan, index=df.index)

    w = w / w.sum()
    mat = np.vstack([df[c].to_numpy(dtype="float64") for c in cols]).T
    num = np.nansum(mat * w, axis=1)
    den = np.nansum((~np.isnan(mat)) * w, axis=1)
    return pd.Series(np.where(den > 0, num / den, np.nan), index=df.index)


def filter_by_position_group(df: pd.DataFrame, group: str) -> pd.DataFrame:
    group = normalize_group(group)
    valid_positions = sb_positions_for(group)
    return df[df["primary_position"].isin(valid_positions)].copy()


# ============= SCORING PRINCIPAL =============
def run_delantero_scoring(
    per90_csv: Path | None = None,
    out_csv: Path | None = None,
    df: pd.DataFrame | None = None,
    position_group: str = "Delantero",
    min_minutes: int = 450,
    min_matches: int = 3,
    flag_q: float = 0.75,
) -> pd.DataFrame:
    """
    Calcula scoring de delanteros usando df o CSV per90.

    Args:
        per90_csv: Path al CSV base (opcional si pasás df)
        out_csv: Path salida (opcional)
        df: DataFrame base (opcional si pasás per90_csv)
        position_group: Grupo de posición ("Delantero")
        min_minutes: minutos mínimos
        min_matches: partidos mínimos
        flag_q: cuantil para flags (0.75 = top 25%)

    Returns:
        DataFrame con scores
    """

    print("=" * 70)
    print(f"SCORING DE {position_group.upper()}")
    print("=" * 70)

    base0 = read_input(per90_csv=per90_csv, df=df)
    print(f"✓ Total jugadores en input: {len(base0):,}")

    # --- Filtrar por posición ---
    print(f"\n🔍 Filtrando por posición: {position_group}")
    base = filter_by_position_group(base0, position_group)
    print(f"✓ Jugadores en posición {position_group}: {len(base):,}")

    # --- Filtrar por minutos y partidos ---
    print(f"\n⏱️  Aplicando filtros:")
    print(f"  - Minutos mínimos: {min_minutes}")
    print(f"  - Partidos mínimos: {min_matches}")

    if "total_minutes" in base.columns:
        base = base[base["total_minutes"] >= min_minutes].copy()
    if "matches_played" in base.columns:
        base = base[base["matches_played"] >= min_matches].copy()

    print(f"✓ Jugadores después de filtros: {len(base):,}")

    if base.empty:
        raise ValueError(f"No hay jugadores de {position_group} que cumplan los filtros.")

    # --- Renombrar columnas para compatibilidad ---
    base = base.rename(columns={
        "teams": "team_name",
        "matches_played": "matches",
        "total_minutes": "minutes",
    })

    # =========================
    # DEFINICIÓN DE CATEGORÍAS
    # =========================
    FINALIZACION = [
        ("xg_per_shot", 0.20, False),
        ("shot_statsbomb_xg_per90", 0.18, False),
        ("obv_total_net_type_shot_per90", 0.15, False),
        ("goals_per90", 0.05, False),
        ("touches_in_opp_box_per90", 0.15, False),
        ("touches_in_opp_box_pct", 0.10, False),
        ("obv_total_net_play_pattern_regular_play_per90", 0.10, False),
        ("total_shots_per90", 0.04, False),
        ("shot_touch_pct", 0.03, False),
    ]

    PRESIONANTE = [
        ("pressure_per90", 0.30, False),
        ("n_events_third_attacking_pressure_per90", 0.20, False),
        ("counterpress_per90", 0.15, False),
        ("ball_recovery_offensive_per90", 0.15, False),
        ("n_events_third_attacking_ball_recovery_per90", 0.10, False),
        ("obv_total_net_type_interception_per90", 0.05, False),
        ("obv_total_net_type_ball_recovery_per90", 0.05, False),
    ]

    CONECTOR = [
        ("complete_passes_per90", 0.25, False),
        ("pass_completion_rate", 0.15, False),
        ("pass_into_final_third_per90", 0.15, False),
        ("obv_total_net_type_pass_per90", 0.25, False),
        ("pass_shot_assist_per90", 0.15, False),
        ("total_turnovers_per90", 0.05, True),
    ]

    DISRUPTIVO = [
        ("obv_total_net_type_dribble_per90", 0.40, False),
        ("obv_total_net_type_carry_per90", 0.35, False),
        ("carry_into_final_third_per90", 0.15, False),
        ("pass_into_final_third_per90", 0.10, False),
    ]

    CATS = {
        "Score_Finalizacion": FINALIZACION,
        "Score_Presionante": PRESIONANTE,
        "Score_Conector": CONECTOR,
        "Score_Disruptivo": DISRUPTIVO,
    }

    CAT_W = {
        "Score_Finalizacion": 0.40,
        "Score_Presionante": 0.10,
        "Score_Conector": 0.25,
        "Score_Disruptivo": 0.25,
    }

    # =========================
    # MÉTRICAS DERIVADAS (solo si faltan)
    # =========================
    print("\n🔧 Calculando métricas derivadas (si aplica)...")

    # complete_passes_per90 (si no existe)
    if "complete_passes_per90" not in base.columns and "complete_passes" in base.columns and "minutes" in base.columns:
        base["complete_passes"] = safe_numeric(base["complete_passes"])
        base["minutes"] = safe_numeric(base["minutes"])
        base["complete_passes_per90"] = np.where(
            base["minutes"] > 0,
            base["complete_passes"] / base["minutes"] * 90.0,
            np.nan,
        )
        print("✓ complete_passes_per90 calculado desde complete_passes")

    # =========================
    # CÁLCULO DE SCORES
    # =========================
    print("\n🎯 Calculando scores...")

    ALL_ITEMS = [item for items in CATS.values() for item in items]

    # Convertir métricas a numérico (todas)
    for col, _, _ in ALL_ITEMS:
        if col in base.columns:
            base[col] = safe_numeric(base[col])

    # Percentiles por métrica
    missing_cols = []
    for col, _, inv in ALL_ITEMS:
        if col not in base.columns:
            missing_cols.append(col)
            continue
        x = -base[col] if inv else base[col]
        base[f"pct__{col}"] = pct_rank_0_100(x)

    if missing_cols:
        print(f"\n⚠️  Columnas no encontradas (serán ignoradas): {len(missing_cols)}")
        for c in missing_cols[:12]:
            print(f"  - {c}")
        if len(missing_cols) > 12:
            print(f"  ... y {len(missing_cols) - 12} más")

    # Score por categoría (promedio ponderado de percentiles)
    for cat, items in CATS.items():
        pct_items = [(f"pct__{col}", w) for col, w, _ in items if f"pct__{col}" in base.columns]
        base[cat] = wavg(base, pct_items) if pct_items else np.nan

    # Overall (promedio ponderado de categorías)
    num = 0.0
    den = 0.0
    for c, w in CAT_W.items():
        if c not in base.columns:
            continue
        valid = base[c].notna()
        num += base[c].fillna(0) * w * valid
        den += w * valid
    base["Score_Overall"] = np.where(den > 0, num / den, np.nan)

    print("✓ Scores calculados")

    # =========================
    # FLAGS Y TAGS
    # =========================
    print(f"\n🏷️  Asignando flags (top {int((1 - flag_q) * 100)}%)...")

    for flag_name, score_col in [
        ("Flag_Finalizacion", "Score_Finalizacion"),
        ("Flag_Presionante", "Score_Presionante"),
        ("Flag_Conector", "Score_Conector"),
        ("Flag_Disruptivo", "Score_Disruptivo"),
    ]:
        if score_col in base.columns and base[score_col].notna().sum() > 0:
            threshold = base[score_col].quantile(flag_q)
            base[flag_name] = base[score_col] >= threshold
        else:
            base[flag_name] = False

    def tags(r):
        t = []
        if r.get("Flag_Finalizacion", False): t.append("Killer")
        if r.get("Flag_Presionante", False): t.append("Presionante")
        if r.get("Flag_Conector", False): t.append("Falso 9")
        if r.get("Flag_Disruptivo", False): t.append("Disruptivo")
        return " | ".join(t) if t else "Balanceados"

    base["Flags"] = base.apply(tags, axis=1)

    # =========================
    # OUTPUT
    # =========================
    cols = [
        "player_id", "player_name", "team_name", "matches", "minutes",
        "primary_position", "primary_position_share",
        "Score_Finalizacion", "Score_Presionante", "Score_Conector", "Score_Disruptivo",
        "Score_Overall",
        "Flag_Finalizacion", "Flag_Presionante", "Flag_Conector", "Flag_Disruptivo",
        "Flags",
    ]
    cols = [c for c in cols if c in base.columns]

    out = base[cols].sort_values("Score_Overall", ascending=False)

    if out_csv is not None:
        out_csv.parent.mkdir(parents=True, exist_ok=True)
        out.to_csv(out_csv, index=False, encoding="utf-8")
        print(f"\n✅ Output guardado en: {out_csv}")

    print("=" * 70)
    print(f"📊 Jugadores evaluados: {len(out):,}")
    return out


# =========================
# EJEMPLO DE USO
# =========================
if __name__ == "__main__":
    per90_csv = Path("outputs/all_players_complete_2025_2026.csv")
    out_csv = Path("outputs/delantero_scores_2025_2026.csv")

    scores = run_delantero_scoring(
        per90_csv=per90_csv,
        out_csv=out_csv,
        position_group="Delantero",
        min_minutes=450,
        min_matches=3,
        flag_q=0.75,
    )
