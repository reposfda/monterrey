# -*- coding: utf-8 -*-
"""
Sistema de Scoring para Interiores / Mediapuntas
Versión simplificada - Solo usa CSV per90 (sin pool_builder, sin cálculo de métricas)

Categorías:
1. Box to Box - Presencia en ambas áreas
2. Desequilibrio / Creativos - Ruptura con regate y conducción
3. Organización - Construcción y progresión con pase
4. Auxilio / Equilibrio - Trabajo defensivo

Requiere:
- all_players_per90_all.csv (output del script principal)
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
        group: Nombre del grupo (ej: "Interior", "Volante")
        
    Returns:
        DataFrame filtrado
    """
    group = normalize_group(group)  # Valida el grupo
    valid_positions = sb_positions_for(group)
    
    # Filtrar por primary_position
    mask = df["primary_position"].isin(valid_positions)
    
    return df[mask].copy()


# ============= SCORING PRINCIPAL =============
def run_interior_scoring(
    per90_csv: Path,
    out_csv: Path,
    position_group: str = "Interior",
    min_minutes: int = 450,
    min_matches: int = 3,
    flag_q: float = 0.75,
):
    """
    Calcula scoring de interiores/mediapuntas usando solo el CSV per90.
    
    Args:
        per90_csv: Path al archivo all_players_per90_all.csv
        out_csv: Path de salida para scores
        position_group: Grupo de posición ("Interior")
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
    per90 = pd.read_csv(per90_csv, low_memory=False, encoding='latin1')
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
    
    # --- 1. BOX TO BOX (presencia en ambas áreas) ---
    BOX_TO_BOX = [
        # Acciones defensivas
        ("interception_success_rate", 0.15, False),
        ("ball_recovery_offensive_per90", 0.10, False),
        ("tackle_success_pct", 0.10, False),
        
        # Acciones ofensivas
        ("shot_statsbomb_xg_per90", 0.15, False),
        ("xa_per90", 0.10, False),
        ("obv_total_net_type_shot_per90", 0.10, False),
        
        # Presencia en área rival
        ("touches_in_opp_box_per90", 0.15, False),
        ("n_events_third_attacking_pass_per90", 0.10, False),
        
        # Volumen general
        ("total_touches_per90", 0.05, False),
    ]

    # --- 2. DESEQUILIBRIO / CREATIVOS (ruptura individual) ---
    DESEQUILIBRIO = [
        # Dribble
        ("obv_total_net_type_dribble_per90", 0.30, False),
        
        # Carry (conducción)
        ("obv_total_net_type_carry_per90", 0.25, False),
        ("carry_into_final_third_per90", 0.15, False),
        
        # Progresiones profundas
        ("pass_into_final_third_per90", 0.10, False),
        
        # Generación de tiros
        ("obv_total_net_type_shot_per90", 0.10, False),
        ("shot_statsbomb_xg_per90", 0.10, False),
    ]

    # --- 3. ORGANIZACIÓN / PROGRESIÓN (construcción con pase) ---
    ORGANIZACION = [
        # OBV de pases
        ("obv_total_net_type_pass_per90", 0.35, False),
        
        # Volumen y precisión
        ("complete_passes_per90", 0.20, False),
        ("pass_completion_rate", 0.10, False),
        
        # Creación
        ("xa_per90", 0.15, False),
        ("obv_total_net_play_pattern_regular_play_per90", 0.10, False),  # xG buildup proxy
        
        # Pérdidas (invertido)
        ("total_turnovers_per90", 0.10, True),
    ]

    # --- 4. AUXILIO / EQUILIBRIO (trabajo defensivo) ---
    AUXILIO = [
        # Recuperaciones en campo rival
        ("ball_recovery_offensive_per90", 0.20, False),
        ("n_events_third_attacking_ball_recovery_per90", 0.10, False),
        
        # Tackles e intercepciones
        ("duel_tackle_per90", 0.15, False),
        ("interception_per90", 0.15, False),
        ("interception_success_rate", 0.05, False),
        
        # Presiones
        ("pressure_per90", 0.15, False),
        ("n_events_third_attacking_pressure_per90", 0.10, False),
        ("counterpress_per90", 0.05, False),
        
        # OBV defensivo
        ("obv_total_net_type_interception_per90", 0.05, False),
    ]

    CATS = {
        "Score_BoxToBox": BOX_TO_BOX,
        "Score_Desequilibrio": DESEQUILIBRIO,
        "Score_Organizacion": ORGANIZACION,
        "Score_Auxilio": AUXILIO,
    }

    # Pesos de categorías para Score_Overall
    CAT_W = {
        "Score_BoxToBox": 0.25,
        "Score_Desequilibrio": 0.30,
        "Score_Organizacion": 0.25,
        "Score_Auxilio": 0.20,
    }
    
    # =========================
    # CALCULAR MÉTRICAS DERIVADAS SI ES NECESARIO
    # =========================
    print("\n🔧 Calculando métricas derivadas...")
    
    # complete_passes_per90 (si no existe)
    if "complete_passes_per90" not in base.columns and "complete_passes" in base.columns:
        base["complete_passes"] = pd.to_numeric(base["complete_passes"], errors="coerce")
        base["complete_passes_per90"] = np.where(
            base["minutes"] > 0,
            base["complete_passes"] / base["minutes"] * 90.0,
            np.nan
        )
        print("✓ complete_passes_per90 calculado")
    
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
        ("Flag_BoxToBox", "Score_BoxToBox"),
        ("Flag_Desequilibrio", "Score_Desequilibrio"),
        ("Flag_Organizacion", "Score_Organizacion"),
        ("Flag_Auxilio", "Score_Auxilio"),
    ]:
        if score_col in base.columns:
            threshold = base[score_col].quantile(flag_q)
            base[flag_name] = base[score_col] >= threshold
        else:
            base[flag_name] = False
    
    # Tags descriptivos
    def tags(r):
        t = []
        if r.get("Flag_BoxToBox", False): t.append("Box to Box")
        if r.get("Flag_Desequilibrio", False): t.append("Desequilibrantes")
        if r.get("Flag_Organizacion", False): t.append("Organizadores")
        if r.get("Flag_Auxilio", False): t.append("Equilibrio")
        return " | ".join(t) if t else "Balanceados"
    
    base["Flags"] = base.apply(tags, axis=1)
    
    # Estadísticas de flags
    flag_counts = {
        "Box to Box": base["Flag_BoxToBox"].sum(),
        "Desequilibrantes": base["Flag_Desequilibrio"].sum(),
        "Organizadores": base["Flag_Organizacion"].sum(),
        "Equilibrio": base["Flag_Auxilio"].sum(),
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
        "Score_BoxToBox", "Score_Desequilibrio", "Score_Organizacion", "Score_Auxilio",
        "Score_Overall",
        "Flag_BoxToBox", "Flag_Desequilibrio", "Flag_Organizacion", "Flag_Auxilio",
        "Flags",
    ]
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
        top5_cols = ["player_name", "team_name", "Score_Overall", "Flags"]
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
    out_csv = Path("outputs/interior_scores_2025_2026.csv")
    
    # Ejecutar scoring para interiores
    scores = run_interior_scoring(
        per90_csv=per90_csv,
        out_csv=out_csv,
        position_group="Interior",
        min_minutes=450,
        min_matches=3,
        flag_q=0.75,  # Top 25%
    )