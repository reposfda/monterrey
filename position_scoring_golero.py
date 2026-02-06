# -*- coding: utf-8 -*-
"""
Sistema de Scoring para Goleros (df-first)

Categorías:
1. Effectiveness
2. Area Domination
3. Foot Play
4. Outside Box

Input:
- df (preferido) o per90_csv (outputs/all_players_complete_2025_2026.csv)
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
def run_goalkeeper_scoring(
    per90_csv: Path | None = None,
    out_csv: Path | None = None,
    df: pd.DataFrame | None = None,
    position_group: str = "Golero",
    min_minutes: int = 450,
    min_matches: int = 3,
    flag_q: float = 0.75,
) -> pd.DataFrame:

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
        raise ValueError(f"No hay {position_group} que cumplan los filtros.")

    # --- Renombrar columnas para compatibilidad ---
    base = base.rename(columns={
        "teams": "team_name",
        "matches_played": "matches",
        "total_minutes": "minutes",
    })

    # =========================
    # DEFINICIÓN DE CATEGORÍAS
    # =========================
    EFFECTIVENESS = [
        ("gk_goals_prevented_per90", 0.50, False),
        ("gk_save_pct", 0.25, False),
        ("gk_errors_leading_to_shot_per90", 0.10, True),
        ("gk_errors_leading_to_goal_per90", 0.15, True),
    ]

    AREA_DOMINATION = [
        ("gk_claims_per90", 0.50, False),
        ("gk_shots_open_play_in_box_against_per90", 0.50, True),
    ]

    FOOT_PLAY = [
        ("gk_pass_obv_per90", 0.40, False),
        ("gk_long_ball_pct", 0.20, False),
        ("gk_pressured_passes_def_third_per90", 0.20, False),
        ("gk_pressured_passes_def_third_completion_pct", 0.20, False),
    ]

    OUTSIDE_BOX = [
        ("gk_actions_outside_box_per90", 0.50, False),
        ("gk_aggressive_distance_avg", 0.50, False),
    ]

    CATS = {
        "Score_Effectiveness": EFFECTIVENESS,
        "Score_Area_Domination": AREA_DOMINATION,
        "Score_Foot_Play": FOOT_PLAY,
        "Score_Outside_Box": OUTSIDE_BOX,
    }

    CAT_W = {
        "Score_Effectiveness": 0.50,
        "Score_Area_Domination": 0.20,
        "Score_Foot_Play": 0.15,
        "Score_Outside_Box": 0.15,
    }

    # =========================
    # CÁLCULO DE SCORES
    # =========================
    print("\n🎯 Calculando scores...")

    ALL_ITEMS = [item for items in CATS.values() for item in items]

    # Convertir métricas a numérico
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

    # Score por categoría
    for cat, items in CATS.items():
        pct_items = [(f"pct__{c}", w) for c, w, _ in items if f"pct__{c}" in base.columns]
        base[cat] = wavg(base, pct_items) if pct_items else np.nan

    # Overall ponderado (robusto a NaNs)
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
    print(f"\n🏷️  Asignando flags (top {int((1-flag_q)*100)}%)...")

    for flag_name, score_col in [
        ("Flag_Effectiveness", "Score_Effectiveness"),
        ("Flag_Area_Domination", "Score_Area_Domination"),
        ("Flag_Foot_Play", "Score_Foot_Play"),
        ("Flag_Outside_Box", "Score_Outside_Box"),
    ]:
        if score_col in base.columns and base[score_col].notna().sum() > 0:
            thr = base[score_col].quantile(flag_q)
            base[flag_name] = base[score_col] >= thr
        else:
            base[flag_name] = False

    def tags(r):
        t = []
        if r.get("Flag_Effectiveness", False): t.append("Atajador")
        if r.get("Flag_Area_Domination", False): t.append("Dominante")
        if r.get("Flag_Foot_Play", False): t.append("Juego de Pies")
        if r.get("Flag_Outside_Box", False): t.append("Líbero")
        return " | ".join(t) if t else "Balanceado"

    base["Flags"] = base.apply(tags, axis=1)

    # =========================
    # OUTPUT
    # =========================
    cols = [
        "player_id", "player_name", "team_name", "matches", "minutes",
        "primary_position", "primary_position_share",
        "Score_Effectiveness", "Score_Area_Domination", "Score_Foot_Play", "Score_Outside_Box",
        "Score_Overall",
        "Flag_Effectiveness", "Flag_Area_Domination", "Flag_Foot_Play", "Flag_Outside_Box",
        "Flags",
    ]
    cols = [c for c in cols if c in base.columns]

    out = base[cols].sort_values("Score_Overall", ascending=False)

    if out_csv is not None:
        out_csv.parent.mkdir(parents=True, exist_ok=True)
        out.to_csv(out_csv, index=False, encoding="utf-8")
        print(f"\n✅ Output guardado en: {out_csv}")

    print("=" * 70)
    print(f"📊 Goleros evaluados: {len(out):,}")
    return out


if __name__ == "__main__":
    per90_csv = Path("outputs/all_players_complete_2025_2026.csv")
    out_csv = Path("outputs/golero_scores_2025_2026.csv")

    scores = run_goalkeeper_scoring(
        per90_csv=per90_csv,
        out_csv=out_csv,
        position_group="Golero",
        min_minutes=450,
        min_matches=3,
        flag_q=0.75,
    )
