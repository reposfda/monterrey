# -*- coding: utf-8 -*-
"""
Sistema de Scoring para Arqueros
Versión adaptada del sistema de scoring de delanteros

Categorías:
1. Effectiveness - Atajadas, goals prevented, errores
2. Area Domination - Claims y tiros en área
3. Foot Play - Juego con los pies
4. Outside Box - Sweeper keeper

Requiere:
- all_players_complete_{season}.csv (output del script principal con métricas de arqueros)
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
        group: Nombre del grupo (ej: "Arquero", "Delantero")
        
    Returns:
        DataFrame filtrado
    """
    group = normalize_group(group)  # Valida el grupo
    valid_positions = sb_positions_for(group)
    
    # Filtrar por primary_position
    mask = df["primary_position"].isin(valid_positions)
    
    return df[mask].copy()


# ============= SCORING PRINCIPAL =============
def run_arquero_scoring(
    complete_csv: Path,
    out_csv: Path,
    position_group: str = "Arquero",
    min_minutes: int = 450,
    min_matches: int = 3,
    flag_q: float = 0.75,
):
    """
    Calcula scoring de arqueros usando el CSV complete.
    
    Args:
        complete_csv: Path al archivo all_players_complete_{season}.csv
        out_csv: Path de salida para scores
        position_group: Grupo de posición ("Arquero")
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
    print(f"\n📂 Cargando: {complete_csv}")
    df = pd.read_csv(complete_csv, low_memory=False, encoding='latin1')
    print(f"✓ Total jugadores en archivo: {len(df):,}")
    
    # --- Filtrar por posición ---
    print(f"\n🔍 Filtrando por posición: {position_group}")
    base = filter_by_position_group(df, position_group)
    print(f"✓ Jugadores en posición {position_group}: {len(base):,}")
    
    # --- Filtrar por minutos y partidos ---
    print(f"\n⏱️  Aplicando filtros:")
    print(f"  - Minutos mínimos: {min_minutes}")
    print(f"  - Partidos mínimos: {min_matches}")
    
    base = base[base["total_minutes"] >= min_minutes].copy()
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
    # Formato: (columna, peso, invertir?)
    # invertir=True para métricas donde menor es mejor
    
    # --- 1. EFFECTIVENESS (Shot Stopping) ---
    EFFECTIVENESS = [
        # Atajadas y prevención
        ("gk_goals_prevented_per90", 0.50, False),  # Métrica principal
        ("gk_save_pct", 0.25, False),
        
        # Errores (invertir: menos es mejor)
        ("gk_errors_leading_to_shot_per90", 0.10, True),
        ("gk_errors_leading_to_goal_per90", 0.15, True),
    ]
    
    # --- 2. AREA DOMINATION ---
    AREA_DOMINATION = [
        # Claims (reclamar balones aéreos)
        ("gk_claims_per90", 0.50, False),
        
        # Tiros recibidos en área (invertir: menos es mejor)
        ("gk_shots_open_play_in_box_against_per90", 0.50, True),
    ]
    
    # --- 3. FOOT PLAY ---
    FOOT_PLAY = [
        # OBV con pases
        ("gk_pass_obv_per90", 0.40, False),
        
        # Pases largos
        ("gk_long_ball_pct", 0.20, False),
        
        # Bajo presión
        ("gk_pressured_passes_def_third_per90", 0.20, False),
        ("gk_pressured_passes_def_third_completion_pct", 0.20, False),
    ]
    
    # --- 4. OUTSIDE BOX (Sweeper Keeper) ---
    OUTSIDE_BOX = [
        # Acciones fuera del área
        ("gk_actions_outside_box_per90", 0.50, False),
        
        # Distancia promedio (más adelantado = mejor)
        ("gk_aggressive_distance_avg", 0.50, False),
    ]
    
    CATS = {
        "Score_Effectiveness": EFFECTIVENESS,
        "Score_Area_Domination": AREA_DOMINATION,
        "Score_Foot_Play": FOOT_PLAY,
        "Score_Outside_Box": OUTSIDE_BOX,
    }
    
    # Pesos de categorías para Score_Overall
    CAT_W = {
        "Score_Effectiveness": 0.50,      # Lo más importante
        "Score_Area_Domination": 0.20,    # Control del área
        "Score_Foot_Play": 0.15,          # Juego con pies
        "Score_Outside_Box": 0.15,        # Sweeper keeper
    }
    
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
        ("Flag_Effectiveness", "Score_Effectiveness"),
        ("Flag_Area_Domination", "Score_Area_Domination"),
        ("Flag_Foot_Play", "Score_Foot_Play"),
        ("Flag_Outside_Box", "Score_Outside_Box"),
    ]:
        if score_col in base.columns:
            threshold = base[score_col].quantile(flag_q)
            base[flag_name] = base[score_col] >= threshold
        else:
            base[flag_name] = False
    
    # Tags descriptivos
    def tags(r):
        t = []
        if r.get("Flag_Effectiveness", False): t.append("Shot Stopper")
        if r.get("Flag_Area_Domination", False): t.append("Dominante")
        if r.get("Flag_Foot_Play", False): t.append("Con Pies")
        if r.get("Flag_Outside_Box", False): t.append("Sweeper")
        return " | ".join(t) if t else "Balanceado"
    
    base["Flags"] = base.apply(tags, axis=1)
    
    # Estadísticas de flags
    flag_counts = {
        "Shot Stopper": base["Flag_Effectiveness"].sum(),
        "Dominante": base["Flag_Area_Domination"].sum(),
        "Con Pies": base["Flag_Foot_Play"].sum(),
        "Sweeper": base["Flag_Outside_Box"].sum(),
    }
    
    print("\n📈 Distribución de flags:")
    for flag, count in flag_counts.items():
        pct = count/len(base)*100 if len(base) > 0 else 0
        print(f"  {flag}: {count} arqueros ({pct:.1f}%)")
    
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
    
    # Crear directorio si no existe
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_csv, index=False, encoding="utf-8")
    
    print("\n✅ SCORING COMPLETADO")
    print("="*70)
    print(f"📁 Output guardado en: {out_csv}")
    print(f"📊 Arqueros evaluados: {len(out):,}")
    
    if not out.empty:
        print(f"\n🏆 Top 5 Arqueros:")
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
    complete_csv = Path("outputs/all_players_complete_2024_2025.csv")
    out_csv = Path("outputs/golero_scores_2024_2025.csv")
    
    # Ejecutar scoring para arqueros
    scores = run_arquero_scoring(
        complete_csv=complete_csv,
        out_csv=out_csv,
        position_group="Golero",
        min_minutes=450,
        min_matches=3,
        flag_q=0.75,  # Top 25%
    )