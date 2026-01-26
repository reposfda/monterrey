# -*- coding: utf-8 -*-
"""
Sistema de Scoring para Extremos
- Misma lógica que position_scoring_interior.py:
  * pct_rank_0_100 por métrica
  * score por categoría = wavg(percentiles, pesos)
  * score overall = wavg(scores categorías, pesos)
  * flags top (1-flag_q)

Requiere:
- per90_csv: all_players_per90_all.csv (o equivalente)
- role_lanes_csv: output del builder extremos OBV carriles (opcional pero recomendado)
"""

from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd
from positions_config import normalize_group, sb_positions_for
from extremos_obv_lanes_builder import build_all_seasons

# =========================
# HELPERS
# =========================
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
    w = w / w.sum()
    mat = np.vstack([df[c].to_numpy(dtype="float64") for c in cols]).T
    num = np.nansum(mat * w, axis=1)
    den = np.nansum((~np.isnan(mat)) * w, axis=1)
    return pd.Series(np.where(den > 0, num / den, np.nan), index=df.index)

def _merge_role_lanes(base: pd.DataFrame, role_lanes_csv: Path | None) -> pd.DataFrame:
    """
    Mergea las métricas del builder de role_lanes (OBV carriles) por player_id (+ season si aplica)
    Espera columnas tipo:
      player_id, _season (opcional), obv_total_pass_per90, obv_from_ext_per90, obv_from_int_per90, lane_bias_origin_index, etc.
    """
    if role_lanes_csv is None:
        return base

    rl = pd.read_csv(role_lanes_csv, low_memory=False)
    # normalizar tipos
    if "player_id" in rl.columns:
        rl["player_id"] = pd.to_numeric(rl["player_id"], errors="coerce")
    if "player_id" in base.columns:
        base["player_id"] = pd.to_numeric(base["player_id"], errors="coerce")

    # merge por season si existe en ambos
    if "_season" in rl.columns and "_season" in base.columns:
        keep = [c for c in rl.columns if c not in ["player_name"]]
        base = base.merge(rl[keep], on=["player_id", "_season"], how="left")
    else:
        keep = [c for c in rl.columns if c not in ["player_name", "_season"]]
        base = base.merge(rl[keep], on="player_id", how="left")

    return base

def lane_tag(r):
    v = r.get("lane_bias_origin_index", np.nan)
    if pd.isna(v):
        return "Sin dato"
    side = "Interior" if v >= 0 else "Exterior"
    strength = r.get("lane_profile_strength", None)
    if pd.isna(strength):
        return side
    return f"{side} ({strength})"


# =========================
# SCORING PRINCIPAL
# =========================
def run_extremos_scoring(
    per90_csv: Path,
    out_csv: Path,
    role_lanes_csv: Path | None = None,
    position_group: str = "Extremo",   # placeholder: si luego lo conectás a positions_config
    min_minutes: int = 450,
    min_matches: int = 3,
    flag_q: float = 0.75,
):
    print("=" * 70)
    print(f"SCORING DE {position_group.upper()}")
    print("=" * 70)

    # =========================
    # BUILD ROLE LANES (si no existe)
    # =========================
    if role_lanes_csv is not None and (not role_lanes_csv.exists()):
        print("\n🛠️  Generando role_lanes de extremos (OBV carriles)...")

        PATHS = {
                "2024-25": Path("outputs/events_2024_2025.csv"),
                "2025-26": Path("outputs/events_2025_2026.csv"),
            }

        build_all_seasons(
            PATHS=PATHS,
            out_dir=str(role_lanes_csv.parent),   # <- string
            minutes_threshold=min_minutes,
            min_share_role=0.60,
        )

        print("✓ role_lanes generado correctamente")
    else:
        print("✓ role_lanes existente, se utiliza el CSV guardado")


    # --- Cargar per90 ---
    print(f"\n📂 Cargando: {per90_csv}")
    base = pd.read_csv(per90_csv, low_memory=False, encoding="latin1")
    print(f"✓ Total jugadores en archivo: {len(base):,}")

    # --- Renombres compatibles (misma idea que interior) ---
    base = base.rename(columns={
        "teams": "team_name",
        "matches_played": "matches",
        "total_minutes": "minutes",
    })

    # --- Filtrar por posición ---
    print(f"\n🔍 Filtrando por posición: {position_group}")
    pos_group = normalize_group(position_group)
    pos_set = set(sb_positions_for(pos_group))

    pos_col = "primary_position"
    if pos_col not in base.columns:
        raise ValueError(f"No existe '{pos_col}' en el dataset. No puedo filtrar por posición.")

    base[pos_col] = base[pos_col].astype(str)
    base = base[base[pos_col].isin(pos_set)].copy()

    print(f"✓ Jugadores filtrados por positions_config ({pos_group}): {len(base):,}")

    # --- Filtros minutos/partidos ---
    if "minutes" in base.columns:
        base["minutes"] = safe_numeric(base["minutes"])
        base = base[base["minutes"] >= min_minutes].copy()
    if "matches" in base.columns:
        base["matches"] = safe_numeric(base["matches"])
        base = base[base["matches"] >= min_matches].copy()

    print(f"✓ Jugadores después de filtros: {len(base):,}")
    if base.empty:
        raise ValueError("No hay extremos que cumplan los filtros.")

    # --- Merge role_lanes (zona de influencia) ---
    base = _merge_role_lanes(base, role_lanes_csv)

    # =========================
    # DEFINICIÓN DE CATEGORÍAS (primer borrador)
    # Formato: (columna, peso, invertir?)
    # =========================

    # 1) COMPROMISO DEFENSIVO
    COMPROMISO_DEF = [
        ("n_events_third_attacking_pressure_per90", 0.20, False),
        ("obv_total_net_third_attacking_pressure_per90", 0.25, False),
        ("counterpress_per90", 0.20, False),
        ("n_events_third_attacking_ball_recovery_per90", 0.15, False),
        ("obv_total_net_type_ball_recovery_per90", 0.20, False),
    ]

    # 2) DESEQUILIBRIO
    DESEQUILIBRIO = [
        ("obv_total_net_type_dribble_per90", 0.18, False),
        ("obv_total_net_type_carry_per90",   0.18, False),

        ("dribble_complete_per90",           0.12, False),  # NUEVA
        ("crosses_completed_per90",          0.10, False),  # NUEVA

        ("obv_total_net_third_attacking_pass_cross_openplay_per90", 0.12, False),
        ("pass_shot_assist_per90",           0.15, False),
        ("pass_goal_assist_per90",           0.05, False),
        ("obv_total_net_type_pass_per90",    0.10, False),
    ]

    # 3) FINALIZACIÓN
    FINALIZACION = [
        ("shot_statsbomb_xg_per90",      0.35, False),
        ("obv_total_net_type_shot_per90",0.25, False),
        ("xg_per_shot",                  0.20, False),          # derivada
        ("touches_in_opp_box_per90",     0.20, False),
    ]

    # 4) ZONA DE INFLUENCIA (role_lanes)
    ZONA_INFLUENCIA = [
        ("obv_total_pass_per90", 0.55, False),
        ("obv_from_ext_per90",   0.225, False),
        ("obv_from_int_per90",   0.225, False),
    ]

    CATS = {
        "Score_CompromisoDefensivo": COMPROMISO_DEF,
        "Score_Desequilibrio": DESEQUILIBRIO,
        "Score_Finalizacion": FINALIZACION,
        "Score_ZonaInfluencia": ZONA_INFLUENCIA,
    }

    # Pesos de categorías para Score_Overall (primer borrador)
    CAT_W = {
        "Score_CompromisoDefensivo": 0.20,
        "Score_Desequilibrio": 0.35,
        "Score_Finalizacion": 0.30,
        "Score_ZonaInfluencia": 0.15,
    }

    # =========================
    # MÉTRICAS DERIVADAS (mínimas)
    # =========================
    print("\n🔧 Calculando métricas derivadas...")

    # xG/shot
    if "xg_per_shot" not in base.columns:
        xg = safe_numeric(base["shot_statsbomb_xg_per90"]) if "shot_statsbomb_xg_per90" in base.columns else np.nan
        shots = safe_numeric(base["shots_per90"]) if "shots_per90" in base.columns else np.nan
        if isinstance(xg, pd.Series) and isinstance(shots, pd.Series):
            base["xg_per_shot"] = np.where(shots > 0, xg / shots, np.nan)
            print("✓ xg_per_shot calculado (desde shot_statsbomb_xg_per90 / shots_per90)")
        else:
            base["xg_per_shot"] = np.nan

    # =========================
    # PERFIL DE INFLUENCIA (inside/outside) - SOLO DESCRIPTOR
    # =========================
    if "lane_bias_origin_index" not in base.columns:
        base["lane_bias_origin_index"] = np.nan

    base["lane_bias_origin_index"] = pd.to_numeric(base["lane_bias_origin_index"], errors="coerce")

    base["lane_influence_side"] = np.where(
        base["lane_bias_origin_index"].isna(),
        "Sin dato",
        np.where(base["lane_bias_origin_index"] >= 0, "Interior", "Exterior")
    )

    base["lane_bias_abs"] = base["lane_bias_origin_index"].abs()

    base["lane_profile_strength"] = pd.cut(
        base["lane_bias_abs"],
        bins=[0.0, 0.15, 0.35, 1.01],
        labels=["Mixto", "Moderado", "Marcado"],
        include_lowest=True
    )

    base["Lane_Profile"] = base.apply(lane_tag, axis=1)



    # =========================
    # PREPARAR NUMÉRICAS + PERCENTILES
    # =========================
    print("\n🎯 Calculando percentiles y scores...")

    # 1) Asegurar numéricas (solo columnas que existan)
    all_metrics = []
    for _, items in CATS.items():
        for col, _, _ in items:
            all_metrics.append(col)

    for col in set(all_metrics):
        if col in base.columns:
            base[col] = safe_numeric(base[col])

    # 2) Percentiles por métrica (invirtiendo si aplica)
    missing_cols = []
    for cat, items in CATS.items():
        for col, _, inv in items:
            if col not in base.columns:
                missing_cols.append(col)
                continue
            x = -base[col] if inv else base[col]
            base[f"pct__{col}"] = pct_rank_0_100(x)

    if missing_cols:
        missing_cols = sorted(set(missing_cols))
        print(f"\n⚠️  Columnas no encontradas (se ignoran): {len(missing_cols)}")
        for c in missing_cols[:15]:
            print(f"  - {c}")
        if len(missing_cols) > 15:
            print(f"  ... y {len(missing_cols) - 15} más")

    # 3) Score por categoría (wavg de percentiles)
    for cat, items in CATS.items():
        pct_items = [(f"pct__{col}", w) for col, w, _ in items if f"pct__{col}" in base.columns]
        if pct_items:
            base[cat] = wavg(base, pct_items)
        else:
            base[cat] = np.nan
            print(f"⚠️  No se pudo calcular {cat} (todas las columnas faltantes)")

    # 4) Overall (wavg de categorías)
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
    # FLAGS
    # =========================
    print(f"\n🏷️  Asignando flags (top {int((1-flag_q)*100)}%)...")

    for flag_name, score_col in [
        ("Flag_CompromisoDef", "Score_CompromisoDefensivo"),
        ("Flag_Desequilibrio", "Score_Desequilibrio"),
        ("Flag_Finalizacion", "Score_Finalizacion"),
        ("Flag_ZonaInfluencia", "Score_ZonaInfluencia"),
    ]:
        if score_col in base.columns:
            thr = base[score_col].quantile(flag_q)
            base[flag_name] = base[score_col] >= thr
        else:
            base[flag_name] = False

    def tags(r):
        t = []
        if r.get("Flag_CompromisoDef", False): t.append("Compromiso Def.")
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
        "Score_CompromisoDefensivo", "Score_Desequilibrio", "Score_Finalizacion", "Score_ZonaInfluencia",
        "Score_Overall",
        "Flag_CompromisoDef", "Flag_Desequilibrio", "Flag_Finalizacion", "Flag_ZonaInfluencia",
        "Flags",
    ]
    cols += ["lane_bias_origin_index", "lane_influence_side", "lane_bias_abs", "Lane_Profile"]
    cols = [c for c in cols if c in base.columns]


    out = base[cols].sort_values("Score_Overall", ascending=False)

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_csv, index=False, encoding="utf-8")
    print("\n✅ SCORING COMPLETADO")
    print("=" * 70)
    print(f"📁 Output guardado en: {out_csv}")
    print(f"📊 Jugadores evaluados: {len(out):,}")

    if not out.empty:
        top_cols = [c for c in ["player_name", "team_name", "Score_Overall", "Flags"] if c in out.columns]
        print("\n🏆 Top 5 Extremos:")
        print(out[top_cols].head().to_string(index=False))

    print("=" * 70)
    return out


if __name__ == "__main__":
    # Ejemplo
    per90_csv = Path("outputs/all_players_per90_all.csv")
    role_lanes_csv = Path("outputs/extremos_obv_origen_destino_ALL_seasons_stacked.csv")
    out_csv = Path("outputs/extremos_scores.csv")

    run_extremos_scoring(
        per90_csv=per90_csv,
        role_lanes_csv=role_lanes_csv,
        out_csv=out_csv,
        position_group="Extremos",
        min_minutes=450,
        min_matches=3,
        flag_q=0.75,
    )
