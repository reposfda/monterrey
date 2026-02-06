# -*- coding: utf-8 -*-
"""
Sistema de Scoring para Extremos
Versión df-first - usa df o CSV per90

Categorías:
1. Compromiso Defensivo
2. Desequilibrio
3. Finalización
4. Zona de Influencia (lane bias: obv_from_ext/int + obv pase)

Requiere:
- all_players_complete_{season}.csv (o df equivalente)
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


def lane_tag(r):
    """Genera tag descriptivo del perfil de carril"""
    v = r.get("lane_bias_index", np.nan)
    if pd.isna(v):
        return "Sin dato"

    side = "Interior" if v >= 0 else "Exterior"

    abs_v = abs(v)
    if abs_v < 0.15:
        strength = "Mixto"
    elif abs_v < 0.35:
        strength = "Moderado"
    else:
        strength = "Marcado"

    return f"{side} ({strength})"


# ============= SCORING PRINCIPAL =============
def run_extremo_scoring(
    per90_csv: Path | None = None,
    out_csv: Path | None = None,
    df: pd.DataFrame | None = None,
    position_group: str = "Extremo",
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
    COMPROMISO_DEF = [
        ("n_events_third_attacking_pressure_per90", 0.20, False),
        ("counterpress_per90", 0.20, False),
        ("n_events_third_attacking_ball_recovery_per90", 0.15, False),
        ("obv_total_net_type_ball_recovery_per90", 0.20, False),
        ("obv_total_net_type_interception_per90", 0.10, False),
        ("pressure_per90", 0.15, False),
    ]

    DESEQUILIBRIO = [
        ("obv_total_net_type_dribble_per90", 0.18, False),
        ("obv_total_net_type_carry_per90", 0.18, False),
        ("carry_into_final_third_per90", 0.10, False),
        ("pass_into_final_third_per90", 0.08, False),
        ("pass_shot_assist_per90", 0.15, False),
        ("pass_goal_assist_per90", 0.05, False),
        ("xa_per90", 0.12, False),
        ("obv_total_net_type_pass_per90", 0.14, False),
    ]

    FINALIZACION = [
        ("shot_statsbomb_xg_per90", 0.35, False),
        ("obv_total_net_type_shot_per90", 0.25, False),
        ("xg_per_shot", 0.20, False),
        ("touches_in_opp_box_per90", 0.20, False),
    ]

    ZONA_INFLUENCIA = [
        ("obv_from_ext_per90", 0.35, False),
        ("obv_from_int_per90", 0.35, False),
        ("obv_total_net_type_pass_per90", 0.30, False),
    ]

    CATS = {
        "Score_CompromisoDef": COMPROMISO_DEF,
        "Score_Desequilibrio": DESEQUILIBRIO,
        "Score_Finalizacion": FINALIZACION,
        "Score_ZonaInfluencia": ZONA_INFLUENCIA,
    }

    CAT_W = {
        "Score_CompromisoDef": 0.20,
        "Score_Desequilibrio": 0.35,
        "Score_Finalizacion": 0.30,
        "Score_ZonaInfluencia": 0.15,
    }

    # =========================
    # MÉTRICAS DERIVADAS (si faltan)
    # =========================
    print("\n🔧 Verificando métricas derivadas...")

    # xg_per_shot
    if "xg_per_shot" not in base.columns:
        print("  ⚠️  xg_per_shot no encontrado, intentando calcular...")
        if "shot_statsbomb_xg_per90" in base.columns and "total_shots_per90" in base.columns:
            base["shot_statsbomb_xg_per90"] = safe_numeric(base["shot_statsbomb_xg_per90"])
            base["total_shots_per90"] = safe_numeric(base["total_shots_per90"])
            base["xg_per_shot"] = np.where(
                base["total_shots_per90"] > 0,
                base["shot_statsbomb_xg_per90"] / base["total_shots_per90"],
                np.nan,
            )
            print("  ✓ xg_per_shot calculado")

    if "xa_per90" not in base.columns:
        print("  ⚠️  xa_per90 no encontrado en el dataset")

    # =========================
    # PERFIL DE CARRIL (descriptivo)
    # =========================
    if "lane_bias_index" in base.columns:
        print("\n🎯 Calculando perfil de influencia por carriles...")
        base["lane_bias_index"] = safe_numeric(base["lane_bias_index"])

        base["lane_influence_side"] = np.where(
            base["lane_bias_index"].isna(),
            "Sin dato",
            np.where(base["lane_bias_index"] >= 0, "Interior", "Exterior"),
        )
        base["Lane_Profile"] = base.apply(lane_tag, axis=1)

        has_bias = int(base["lane_bias_index"].notna().sum())
        if has_bias > 0:
            ext_count = int((base["lane_influence_side"] == "Exterior").sum())
            int_count = int((base["lane_influence_side"] == "Interior").sum())
            print(f"  ✓ Jugadores con perfil de carril: {has_bias}")
            print(f"    - Perfil Exterior: {ext_count}")
            print(f"    - Perfil Interior: {int_count}")
    else:
        print("\n⚠️  lane_bias_index no encontrado (métricas de carriles no disponibles)")
        base["lane_influence_side"] = "Sin dato"
        base["Lane_Profile"] = "Sin dato"

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

    # Overall ponderado por categorías (robusto a NaNs)
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
        ("Flag_CompromisoDef", "Score_CompromisoDef"),
        ("Flag_Desequilibrio", "Score_Desequilibrio"),
        ("Flag_Finalizacion", "Score_Finalizacion"),
        ("Flag_ZonaInfluencia", "Score_ZonaInfluencia"),
    ]:
        if score_col in base.columns and base[score_col].notna().sum() > 0:
            thr = base[score_col].quantile(flag_q)
            base[flag_name] = base[score_col] >= thr
        else:
            base[flag_name] = False

    def tags(r):
        t = []
        if r.get("Flag_CompromisoDef", False): t.append("Compromiso Def")
        if r.get("Flag_Desequilibrio", False): t.append("Desequilibrio")
        if r.get("Flag_Finalizacion", False): t.append("Finalización")
        if r.get("Flag_ZonaInfluencia", False): t.append("Zona Influencia")
        return " | ".join(t) if t else "Balanceados"

    base["Flags"] = base.apply(tags, axis=1)

    # =========================
    # OUTPUT
    # =========================
    cols = [
        "player_id", "player_name", "team_name", "matches", "minutes",
        "primary_position", "primary_position_share",
        "Score_CompromisoDef", "Score_Desequilibrio", "Score_Finalizacion", "Score_ZonaInfluencia",
        "Score_Overall",
        "Flag_CompromisoDef", "Flag_Desequilibrio", "Flag_Finalizacion", "Flag_ZonaInfluencia",
        "Flags",
    ]

    # Agregar columnas de perfil de carril si existen
    for c in ["lane_bias_index", "lane_influence_side", "Lane_Profile"]:
        if c in base.columns and c not in cols:
            cols.append(c)

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
    out_csv = Path("outputs/extremo_scores_2025_2026.csv")

    scores = run_extremo_scoring(
        per90_csv=per90_csv,
        out_csv=out_csv,
        position_group="Extremo",
        min_minutes=450,
        min_matches=3,
        flag_q=0.75,
    )
