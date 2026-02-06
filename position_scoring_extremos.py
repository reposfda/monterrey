# -*- coding: utf-8 -*-
"""
Sistema de Scoring para Extremos
Versión simplificada - Solo usa CSV per90 (sin pool_builder, sin cálculo de métricas)

Categorías:
1. Compromiso Defensivo - Presión y recuperaciones altas
2. Desequilibrio - Dribble, carry, asistencias
3. Finalización - xG, shots, toques en área
4. Zona de Influencia - OBV desde exterior vs interior (lane bias)

Requiere:
- all_players_complete_{season}.csv (output del script principal con métricas de carriles)
- positions_config.py (módulo de configuración)
"""

from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd
from positions_config import normalize_group, sb_positions_for


# ============= HELPERS =============
def safe_numeric(s: pd.Series) -> pd.Series:
    """Convierte a numérico de forma segura"""
    return pd.to_numeric(s, errors="coerce")


def pct_rank_0_100(s: pd.Series) -> pd.Series:
    """
    Calcula percentil de 0 a 100.
    Valores más altos = mejor posición (mayor percentil).
    """
    x = s.copy()
    m = x.notna()
    out = pd.Series(np.nan, index=x.index, dtype="float64")
    out.loc[m] = x.loc[m].rank(pct=True, method="average") * 100.0
    return out


def wavg(df: pd.DataFrame, cols_weights):
    """
    Calcula promedio ponderado ignorando NaN.
    
    Args:
        df: DataFrame con las columnas
        cols_weights: Lista de tuplas (columna, peso)
        
    Returns:
        Serie con el promedio ponderado
    """
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
    """
    Filtra jugadores por grupo de posición usando primary_position.
    
    Args:
        df: DataFrame con columna 'primary_position'
        group: Nombre del grupo (ej: "Extremo")
        
    Returns:
        DataFrame filtrado
    """
    group = normalize_group(group)  # Valida el grupo
    valid_positions = sb_positions_for(group)
    
    # Filtrar por primary_position
    mask = df["primary_position"].isin(valid_positions)
    
    return df[mask].copy()


def lane_tag(r):
    """Genera tag descriptivo del perfil de carril"""
    v = r.get("lane_bias_index", np.nan)
    if pd.isna(v):
        return "Sin dato"
    
    side = "Interior" if v >= 0 else "Exterior"
    
    # Clasificar fuerza del sesgo
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
    per90_csv: Path,
    out_csv: Path,
    position_group: str = "Extremo",
    min_minutes: int = 450,
    min_matches: int = 3,
    flag_q: float = 0.75,
):
    """
    Calcula scoring de extremos usando solo el CSV per90.
    
    Args:
        per90_csv: Path al archivo all_players_complete.csv
        out_csv: Path de salida para scores
        position_group: Grupo de posición ("Extremo")
        min_minutes: Minutos mínimos requeridos
        min_matches: Partidos mínimos requeridos
        flag_q: Cuantil para flags (0.75 = top 25%)
        
    Returns:
        DataFrame con los scores calculados
    """
    
    print("="*70)
    print(f"SCORING DE {position_group.upper()}")
    print("="*70)
    
    # --- Cargar datos ---
    print(f"\n📂 Cargando: {per90_csv}")
    per90 = pd.read_csv(per90_csv, low_memory=False, encoding='utf-8-sig')
    print(f"✓ Total jugadores en archivo: {len(per90):,}")
    
    # --- Filtrar por posición ---
    print(f"\n🔍 Filtrando por posición: {position_group}")
    base = filter_by_position_group(per90, position_group)
    print(f"✓ Jugadores en posición {position_group}: {len(base):,}")
    
    # --- Filtrar por minutos y partidos ---
    print(f"\n⏱️  Aplicando filtros:")
    print(f"  - Minutos mínimos: {min_minutes}")
    print(f"  - Partidos mínimos: {min_matches}")
    
    base = base[base["total_minutes"] >= min_minutes].copy()
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
    # Formato: (columna, peso, invertir?)
    # invertir=True para métricas donde menor es mejor
    
    # --- 1. COMPROMISO DEFENSIVO ---
    COMPROMISO_DEF = [
        ("n_events_third_attacking_pressure_per90", 0.20, False),
        ("counterpress_per90", 0.20, False),
        ("n_events_third_attacking_ball_recovery_per90", 0.15, False),
        ("obv_total_net_type_ball_recovery_per90", 0.20, False),
        ("obv_total_net_type_interception_per90", 0.10, False),
        ("pressure_per90", 0.15, False),
    ]

    # --- 2. DESEQUILIBRIO ---
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

    # --- 3. FINALIZACIÓN ---
    FINALIZACION = [
        ("shot_statsbomb_xg_per90", 0.35, False),
        ("obv_total_net_type_shot_per90", 0.25, False),
        ("xg_per_shot", 0.20, False),
        ("touches_in_opp_box_per90", 0.20, False),
    ]

    # --- 4. ZONA DE INFLUENCIA (métricas de carriles) ---
    ZONA_INFLUENCIA = [
        ("obv_from_ext_per90", 0.35, False),  # OBV desde bandas
        ("obv_from_int_per90", 0.35, False),  # OBV desde interior
        ("obv_total_net_type_pass_per90", 0.30, False),  # OBV total de pases
    ]

    CATS = {
        "Score_CompromisoDef": COMPROMISO_DEF,
        "Score_Desequilibrio": DESEQUILIBRIO,
        "Score_Finalizacion": FINALIZACION,
        "Score_ZonaInfluencia": ZONA_INFLUENCIA,
    }

    # Pesos de categorías para Score_Overall
    CAT_W = {
        "Score_CompromisoDef": 0.20,
        "Score_Desequilibrio": 0.35,
        "Score_Finalizacion": 0.30,
        "Score_ZonaInfluencia": 0.15,
    }
    
    # =========================
    # CALCULAR MÉTRICAS DERIVADAS SI ES NECESARIO
    # =========================
    print("\n🔧 Verificando métricas derivadas...")
    
    # xg_per_shot (ya debería estar calculado en main_analysis)
    if "xg_per_shot" not in base.columns:
        print("  ⚠️  xg_per_shot no encontrado, intentando calcular...")
        if "shot_statsbomb_xg_per90" in base.columns and "total_shots_per90" in base.columns:
            base["xg_per_shot"] = np.where(
                base["total_shots_per90"] > 0,
                base["shot_statsbomb_xg_per90"] / base["total_shots_per90"],
                np.nan
            )
            print("  ✓ xg_per_shot calculado")
    
    # xa_per90 (debería venir del dataset)
    if "xa_per90" not in base.columns:
        print("  ⚠️  xa_per90 no encontrado en el dataset")
    
    # =========================
    # PERFIL DE INFLUENCIA (DESCRIPTIVO)
    # =========================
    if "lane_bias_index" in base.columns:
        print("\n🎯 Calculando perfil de influencia por carriles...")
        
        base["lane_bias_index"] = pd.to_numeric(base["lane_bias_index"], errors="coerce")
        
        base["lane_influence_side"] = np.where(
            base["lane_bias_index"].isna(),
            "Sin dato",
            np.where(base["lane_bias_index"] >= 0, "Interior", "Exterior")
        )
        
        base["Lane_Profile"] = base.apply(lane_tag, axis=1)
        
        # Estadísticas
        has_bias = base["lane_bias_index"].notna().sum()
        if has_bias > 0:
            ext_count = (base["lane_influence_side"] == "Exterior").sum()
            int_count = (base["lane_influence_side"] == "Interior").sum()
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
    
    # Convertir métricas a numérico
    all_metrics = []
    for cat, items in CATS.items():
        for col, _, _ in items:
            all_metrics.append(col)
    
    for col in set(all_metrics):
        if col in base.columns:
            base[col] = safe_numeric(base[col])
    
    # Percentiles por métrica
    missing_cols = []
    for cat, items in CATS.items():
        for col, _, inv in items:
            if col not in base.columns:
                missing_cols.append(col)
                continue
            
            # Invertir si es necesario (menor valor = mejor)
            x = -base[col] if inv else base[col]
            base[f"pct__{col}"] = pct_rank_0_100(x)
    
    if missing_cols:
        print(f"\n⚠️  Columnas no encontradas (serán ignoradas): {len(missing_cols)}")
        for col in missing_cols[:10]:  # Mostrar solo las primeras 10
            print(f"  - {col}")
        if len(missing_cols) > 10:
            print(f"  ... y {len(missing_cols) - 10} más")
    
    # Score por categoría (promedio ponderado de percentiles)
    for cat, items in CATS.items():
        pct_items = [(f"pct__{col}", w) for col, w, _ in items if f"pct__{col}" in base.columns]
        if pct_items:
            base[cat] = wavg(base, pct_items)
        else:
            base[cat] = np.nan
            print(f"⚠️  No se pudo calcular {cat} (todas las columnas faltantes)")
    
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
    print(f"\n🏷️  Asignando flags (top {int((1-flag_q)*100)}%)...")
    
    # Flags basados en cuantil
    for flag_name, score_col in [
        ("Flag_CompromisoDef", "Score_CompromisoDef"),
        ("Flag_Desequilibrio", "Score_Desequilibrio"),
        ("Flag_Finalizacion", "Score_Finalizacion"),
        ("Flag_ZonaInfluencia", "Score_ZonaInfluencia"),
    ]:
        if score_col in base.columns:
            threshold = base[score_col].quantile(flag_q)
            base[flag_name] = base[score_col] >= threshold
        else:
            base[flag_name] = False
    
    # Tags descriptivos
    def tags(r):
        t = []
        if r.get("Flag_CompromisoDef", False): t.append("Compromiso Def")
        if r.get("Flag_Desequilibrio", False): t.append("Desequilibrio")
        if r.get("Flag_Finalizacion", False): t.append("Finalización")
        if r.get("Flag_ZonaInfluencia", False): t.append("Zona Influencia")
        return " | ".join(t) if t else "Balanceados"
    
    base["Flags"] = base.apply(tags, axis=1)
    
    # Estadísticas de flags
    flag_counts = {
        "Compromiso Def": base["Flag_CompromisoDef"].sum(),
        "Desequilibrio": base["Flag_Desequilibrio"].sum(),
        "Finalización": base["Flag_Finalizacion"].sum(),
        "Zona Influencia": base["Flag_ZonaInfluencia"].sum(),
    }
    
    print("\n📈 Distribución de flags:")
    for flag, count in flag_counts.items():
        pct = count/len(base)*100 if len(base) > 0 else 0
        print(f"  {flag}: {count} jugadores ({pct:.1f}%)")
    
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
    for col in ["lane_bias_index", "lane_influence_side", "Lane_Profile"]:
        if col in base.columns:
            cols.append(col)
    
    cols = [c for c in cols if c in base.columns]
    
    out = base[cols].sort_values("Score_Overall", ascending=False)
    
    # Crear directorio si no existe
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_csv, index=False, encoding="utf-8")
    
    print("\n✅ SCORING COMPLETADO")
    print("="*70)
    print(f"📁 Output guardado en: {out_csv}")
    print(f"📊 Jugadores evaluados: {len(out):,}")
    
    if not out.empty:
        print(f"\n🏆 Top 5 {position_group}:")
        top5_cols = ["player_name", "team_name", "Score_Overall", "Flags", "Lane_Profile"]
        top5_cols = [c for c in top5_cols if c in out.columns]
        print(out[top5_cols].head().to_string(index=False))
    
    print("="*70)
    
    return out


# =========================
# EJEMPLO DE USO
# =========================
if __name__ == "__main__":
    from pathlib import Path
    
    # Rutas
    per90_csv = Path("outputs/all_players_complete_2025_2026.csv")
    out_csv = Path("outputs/extremo_scores_2025_2026.csv")
    
    # Ejecutar scoring para extremos
    scores = run_extremo_scoring(
        per90_csv=per90_csv,
        out_csv=out_csv,
        position_group="Extremo",
        min_minutes=450,
        min_matches=3,
        flag_q=0.75,  # Top 25%
    )