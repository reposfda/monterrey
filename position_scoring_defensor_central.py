# -*- coding: utf-8 -*-
"""
Sistema de Scoring para Defensores Centrales (CB)
Adaptado al pipeline per90 (estilo interiores)

- Usa CSV per90
- Filtra por posición, minutos y partidos
- Calcula percentiles, scores, flags y tags
"""

from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd
from positions_config import normalize_group, sb_positions_for


# =========================
# HELPERS
# =========================
def safe_numeric(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce")


def pct_rank_0_100(s: pd.Series) -> pd.Series:
    out = pd.Series(np.nan, index=s.index, dtype="float64")
    m = s.notna()
    out.loc[m] = s.loc[m].rank(pct=True, method="average") * 100.0
    return out


def wavg(df: pd.DataFrame, cols_weights):
    cols = [c for c, _ in cols_weights if c in df.columns]
    if not cols:
        return pd.Series(np.nan, index=df.index)

    w = np.array([w for c, w in cols_weights if c in df.columns], dtype="float64")
    w = w / w.sum()
    mat = np.vstack([df[c].to_numpy(dtype="float64") for c in cols]).T
    num = np.nansum(mat * w, axis=1)
    den = np.nansum((~np.isnan(mat)) * w, axis=1)
    return pd.Series(np.where(den > 0, num / den, np.nan), index=df.index)


def filter_by_position_group(df: pd.DataFrame, group: str) -> pd.DataFrame:
    group = normalize_group(group)
    valid_positions = sb_positions_for(group)
    return df[df["primary_position"].isin(valid_positions)].copy()


# =========================
# DEFINICIÓN DE MÉTRICAS (CB)
# =========================
ACCION_DEF = [
    ("duel_success_rate", 0.25, False),
    ("tackle_success_pct", 0.25, False),
    ("ball_recovery_success_pct", 0.15, False),
    ("interception_success_rate", 0.10, False),
    ("clearances_total_per90", 0.10, False),
    ("blocks_total_per90", 0.05, False),
    ("defensive_actions_lost_per90", 0.10, True),
]

CONTROL_DEF = [
    ("pressure_per90",        0.15, False),
    ("counterpress_per90",    0.15, False),
    ("obv_into_per90",        0.25, True),
    ("obv_from_per90",        0.25, True),
    ("shots_from_area_per90", 0.10, True),
    ("xg_from_area_per90",    0.10, True),
]

PROGRESION = [
    ("pass_completion_rate",          0.15, False),
    ("pass_into_final_third_per90",   0.20, False),
    ("pass_switch_per90",             0.20, False),
    ("carry_into_final_third_per90",  0.10, False),
    ("pass_through_ball_per90",       0.15, False),
    ("obv_total_net_type_pass_per90", 0.20, False),
]

OFENSIVO = [
    ("shot_statsbomb_xg_play_pattern_regular_play_per90",   0.15, False),
    ("shot_statsbomb_xg_play_pattern_from_corner_per90",    0.075, False),
    ("shot_statsbomb_xg_play_pattern_from_free_kick_per90", 0.075, False),
    ("obv_total_net_play_pattern_regular_play_per90",       0.15, False),
    ("obv_total_net_play_pattern_from_free_kick_per90",     0.075, False),
    ("obv_total_net_play_pattern_from_corner_per90",        0.075, False),
]

CATS = {
    "Score_AccionDefensiva": ACCION_DEF,
    "Score_ControlDefensivo": CONTROL_DEF,
    "Score_Progresion": PROGRESION,
    "Score_ImpactoOfensivo": OFENSIVO,
}

CAT_W = {
    "Score_AccionDefensiva": 0.25,
    "Score_ControlDefensivo": 0.45,
    "Score_Progresion": 0.20,
    "Score_ImpactoOfensivo": 0.10,
}


# =========================
# SCORING PRINCIPAL
# =========================
def run_cb_scoring(
    per90_csv: Path,
    out_csv: Path,
    position_group: str = "Zaguero",
    min_minutes: int = 600,
    min_matches: int = 5,
    flag_q: float = 0.75,
):

    print("=" * 70)
    print("SCORING DEFENSORES CENTRALES")
    print("=" * 70)

    df = pd.read_csv(per90_csv, low_memory=False)
    print(f"✓ Jugadores totales: {len(df):,}")

    # --- Filtro posición ---
    df = filter_by_position_group(df, position_group)
    print(f"✓ Centrales: {len(df):,}")

    # --- Filtro minutos / partidos ---
    df = df[df["total_minutes"] >= min_minutes]
    df = df[df["matches_played"] >= min_matches]
    print(f"✓ Tras filtros: {len(df):,}")

    if df.empty:
        raise ValueError("No hay centrales que cumplan los filtros.")

    # Renombres
    df = df.rename(columns={
        "teams": "team_name",
        "matches_played": "matches",
        "total_minutes": "minutes",
    })

    # --- Numeric ---
    all_metrics = {c for items in CATS.values() for c, _, _ in items}
    for col in all_metrics:
        if col in df.columns:
            df[col] = safe_numeric(df[col])

    # --- Percentiles ---
    for items in CATS.values():
        for col, _, inv in items:
            if col not in df.columns:
                continue
            x = -df[col] if inv else df[col]
            df[f"pct__{col}"] = pct_rank_0_100(x)

    # --- Scores por categoría ---
    for cat, items in CATS.items():
        pct_items = [(f"pct__{c}", w) for c, w, _ in items if f"pct__{c}" in df.columns]
        df[cat] = wavg(df, pct_items)

    # --- Score final ---
    df["Score_Overall"] = wavg(df, list(CAT_W.items()))

    # --- Flags ---
    for cat in CATS.keys():
        threshold = df[cat].quantile(flag_q)
        df[f"Flag_{cat.replace('Score_', '')}"] = df[cat] >= threshold

    # --- Tags ---
    def tags(r):
        t = []
        if r["Flag_AccionDefensiva"]: t.append("Acción Def")
        if r["Flag_ControlDefensivo"]: t.append("Control Def")
        if r["Flag_Progresion"]: t.append("Progresión")
        if r["Flag_ImpactoOfensivo"]: t.append("Ofensivo")
        return " | ".join(t) if t else "Balanceado"

    df["Flags"] = df.apply(tags, axis=1)

    # --- Output ---
    cols = [
        "player_name", "team_name", "minutes", "matches",
        "Score_Overall",
        *CATS.keys(),
        "Flags",
    ]
    cols = [c for c in cols if c in df.columns]

    out = df[cols].sort_values("Score_Overall", ascending=False)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_csv, index=False)

    print(f"✅ Output guardado en {out_csv}")
    print("=" * 70)

    return out

# =========================
# EJEMPLO DE USO
# =========================
if __name__ == "__main__":
    from pathlib import Path
    
    # Rutas
    per90_csv = Path("outputs/all_players_complete_2025_2026.csv")
    out_csv = Path("outputs/zaguero_scores_2025_2026.csv")
    
    # Ejecutar scoring para zagueros
    scores = run_cb_scoring(
        per90_csv=per90_csv,
        out_csv=out_csv,
        position_group="Zaguero",
        min_minutes=450,
        min_matches=3,
        flag_q=0.75,  # Top 25%
    )