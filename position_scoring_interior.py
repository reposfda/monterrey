# -*- coding: utf-8 -*-
"""
Sistema de Scoring para Interiores / Mediapuntas (df-first)

Categorías:
1. Box to Box
2. Desequilibrio
3. Organización
4. Contención / Presión

Input:
- df (preferido) o per90_csv
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
def run_interior_scoring(
    per90_csv: Path | None = None,
    out_csv: Path | None = None,
    df: pd.DataFrame | None = None,
    position_group: str = "Interior/Mediapunta",
    min_minutes: int = 450,
    min_matches: int = 3,
    flag_q: float = 0.75,
) -> pd.DataFrame:

    print("=" * 70)
    print(f"SCORING DE {position_group.upper()}")
    print("=" * 70)

    per90 = read_input(per90_csv=per90_csv, df=df)
    print(f"✓ Total jugadores en input: {len(per90):,}")

    # --- Filtrar por posición ---
    print(f"\n🔍 Filtrando por posición: {position_group}")
    base = filter_by_position_group(per90, position_group)
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
    BOX_TO_BOX = [
        ("n_events_third_defensive_ball_recovery_per90", 0.11, False),
        ("n_events_third_middle_ball_recovery_per90",    0.14, False),
        ("n_events_third_attacking_ball_recovery_per90", 0.11, False),

        ("n_events_third_defensive_duel_per90", 0.08, False),
        ("n_events_third_middle_duel_per90",    0.14, False),
        ("n_events_third_attacking_duel_per90", 0.08, False),

        ("carry_into_final_third_per90", 0.10, False),
        ("touches_in_opp_box_per90",     0.10, False),
        ("shot_touch_pct",               0.05, False),

        ("total_touches_per90", 0.09, False),
    ]

    DESEQUILIBRIO = [
        ("obv_total_net_type_dribble_per90", 0.30, False),
        ("obv_total_net_type_carry_per90",   0.25, False),

        ("carry_into_final_third_per90", 0.15, False),
        ("pass_into_final_third_per90",  0.10, False),

        ("obv_total_net_type_shot_per90", 0.10, False),
        ("shot_statsbomb_xg_per90",       0.10, False),
    ]

    ORGANIZACION = [
        ("obv_total_net_type_pass_per90", 0.30, False),
        ("complete_passes_per90",         0.20, False),

        ("pass_shot_assist_per90",                   0.12, False),
        ("obv_total_net_third_attacking_pass_per90", 0.13, False),
        ("obv_total_net_play_pattern_regular_play_per90", 0.10, False),

        ("total_turnovers_per90", 0.15, True),
    ]

    CONTENCION_PRESION = [
        ("n_events_third_middle_pressure_per90",     0.18, False),
        ("n_events_third_attacking_pressure_per90",  0.12, False),
        ("counterpress_per90",                      0.10, False),

        ("n_events_third_middle_ball_recovery_per90",    0.12, False),
        ("n_events_third_attacking_ball_recovery_per90", 0.13, False),

        ("obv_total_net_duel_type_tackle_per90", 0.10, False),
        ("duel_tackle_per90",                    0.10, False),

        ("obv_total_net_type_interception_per90",        0.08, False),
        ("obv_total_net_third_middle_interception_per90", 0.07, False),
    ]

    CATS = {
        "Score_BoxToBox": BOX_TO_BOX,
        "Score_Desequilibrio": DESEQUILIBRIO,
        "Score_Organizacion": ORGANIZACION,
        "Score_ContencionPresion": CONTENCION_PRESION,
    }

    CAT_W = {
        "Score_BoxToBox": 0.25,
        "Score_Desequilibrio": 0.30,
        "Score_Organizacion": 0.25,
        "Score_ContencionPresion": 0.20,
    }

    # =========================
    # MÉTRICAS DERIVADAS (si aplica)
    # =========================
    print("\n🔧 Verificando métricas derivadas...")

    if "complete_passes_per90" not in base.columns and "complete_passes" in base.columns and "minutes" in base.columns:
        base["complete_passes"] = safe_numeric(base["complete_passes"])
        base["complete_passes_per90"] = np.where(
            safe_numeric(base["minutes"]) > 0,
            base["complete_passes"] / safe_numeric(base["minutes"]) * 90.0,
            np.nan
        )
        print("✓ complete_passes_per90 calculado")

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

    missing_cols = sorted(set(missing_cols))
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

    # Overall ponderado robusto
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
        ("Flag_BoxToBox", "Score_BoxToBox"),
        ("Flag_Desequilibrio", "Score_Desequilibrio"),
        ("Flag_Organizacion", "Score_Organizacion"),
        ("Flag_ContencionPresion", "Score_ContencionPresion"),
    ]:
        if score_col in base.columns and base[score_col].notna().sum() > 0:
            thr = base[score_col].quantile(flag_q)
            base[flag_name] = base[score_col] >= thr
        else:
            base[flag_name] = False

    def tags(r):
        t = []
        if r.get("Flag_BoxToBox", False): t.append("Box to Box")
        if r.get("Flag_Desequilibrio", False): t.append("Desequilibrantes")
        if r.get("Flag_Organizacion", False): t.append("Organizadores")
        if r.get("Flag_ContencionPresion", False): t.append("Contención/Presión")
        return " | ".join(t) if t else "Balanceados"

    base["Flags"] = base.apply(tags, axis=1)

    # =========================
    # OUTPUT
    # =========================
    cols = [
        "player_id", "player_name", "team_name", "matches", "minutes",
        "primary_position", "primary_position_share",
        "Score_BoxToBox", "Score_Desequilibrio", "Score_Organizacion", "Score_ContencionPresion",
        "Score_Overall",
        "Flag_BoxToBox", "Flag_Desequilibrio", "Flag_Organizacion", "Flag_ContencionPresion",
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


if __name__ == "__main__":
    per90_csv = Path("outputs/all_players_complete_2025_2026.csv")
    out_csv = Path("outputs/interior_scores_2025_2026.csv")

    scores = run_interior_scoring(
        per90_csv=per90_csv,
        out_csv=out_csv,
        position_group="Interior/Mediapunta",
        min_minutes=450,
        min_matches=3,
        flag_q=0.75,
    )
