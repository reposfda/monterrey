# -*- coding: utf-8 -*-
"""
obv_lanes_builder.py
Calcula índice de sesgo de carriles (lane bias) para TODOS los jugadores.

Clasifica pases según carril de origen (exterior/interior) y calcula:
- OBV desde exterior vs interior
- Índice de sesgo: positivo = prefiere interior, negativo = prefiere exterior

IMPORTANTE: 
- Se llama DESDE main_analysis.py
- Reutiliza datos ya calculados (no duplica lógica)
- Retorna DataFrame para merge con all_players_complete
"""

import numpy as np
import pandas as pd


def lane_id_from_y(y, pitch_width):
    """
    Clasifica coordenada Y en uno de 5 carriles.
    
    Carriles (para pitch de 80m):
    1: 0-16m   (banda izquierda)
    2: 16-32m  (interior izquierdo)
    3: 32-48m  (centro)
    4: 48-64m  (interior derecho)
    5: 64-80m  (banda derecha)
    """
    if pd.isna(y):
        return np.nan
    y = float(y)
    y = max(0.0, min(pitch_width, y))
    step = pitch_width / 5.0
    lane = int(y // step) + 1
    return min(5, max(1, lane))


def lane_group_2(lane):
    """
    Agrupa 5 carriles en 2 zonas.
    
    ext (exterior): Carriles 1 y 5 (bandas)
    int (interior): Carriles 2, 3, 4 (zona central)
    """
    if pd.isna(lane):
        return np.nan
    lane = int(lane)
    return "ext" if lane in (1, 5) else "int"


def calculate_lane_bias_metrics(
    df: pd.DataFrame,
    player_minutes_summary: pd.DataFrame,
    pitch_width: float,
    extract_x_y_func=None,
    min_passes: int = 50,
) -> pd.DataFrame:
    """
    Calcula métricas de OBV por carriles para TODOS los jugadores.
    
    Args:
        df: DataFrame de eventos (con 'pid' ya calculado)
        player_minutes_summary: DF con player_id, total_minutes
        pitch_width: Ancho de cancha (ya inferido)
        extract_x_y_func: Función extract_x_y_from_location del main_analysis
        min_passes: Mínimo de pases para calcular bias (default: 50)
        
    Returns:
        DataFrame con columnas:
        - player_id
        - obv_from_ext_per90: OBV per90 desde exterior
        - obv_from_int_per90: OBV per90 desde interior
        - lane_bias_index: Índice de sesgo (-1 a +1)
        - total_passes_for_lanes: Total de pases usados
    """
    
    print("\n--- Calculando métricas de carriles para todos los jugadores ---")
    
    # === PASO 1: Filtrar pases ===
    df_pass = df[df["type"] == "Pass"].copy()
    
    if df_pass.empty or "pass_end_location" not in df_pass.columns:
        print("  ⚠️  Sin pases válidos")
        return pd.DataFrame()
    
    # === PASO 2: Extraer coordenadas Y ===
    df_pass["start_y"] = pd.to_numeric(df_pass["y"], errors="coerce")
    
    # Y final (usando función del main_analysis si está disponible)
    if extract_x_y_func:
        df_pass["end_x"], df_pass["end_y"] = zip(
            *df_pass["pass_end_location"].apply(extract_x_y_func)
        )
    else:
        # Fallback simple
        def get_y(val):
            if pd.isna(val):
                return np.nan
            if isinstance(val, str):
                try:
                    import json
                    val = json.loads(val)
                except:
                    return np.nan
            if isinstance(val, (list, tuple)) and len(val) >= 2:
                return float(val[1])
            return np.nan
        
        df_pass["end_y"] = df_pass["pass_end_location"].apply(get_y)
    
    df_pass = df_pass.dropna(subset=["start_y", "end_y", "pid"])
    
    if df_pass.empty:
        print("  ⚠️  Sin coordenadas válidas")
        return pd.DataFrame()
    
    print(f"  Total pases válidos: {len(df_pass):,}")
    
    # === PASO 3: Clasificar en carriles ===
    df_pass["start_lane"] = df_pass["start_y"].apply(lambda y: lane_id_from_y(y, pitch_width))
    df_pass["start_group"] = df_pass["start_lane"].apply(lane_group_2)
    df_pass["obv"] = pd.to_numeric(df_pass["obv_total_net"], errors="coerce").fillna(0.0)
    
    # === PASO 4: Agregar por jugador y grupo de origen ===
    # Sumar OBV por jugador y grupo
    obv_by_group = df_pass.groupby(["pid", "start_group"], as_index=False).agg({
        "obv": "sum"
    }).rename(columns={"obv": "obv_total"})
    
    # Contar pases por jugador (separado para evitar perder 'pid')
    passes_by_player = df_pass.groupby("pid", as_index=False).size().rename(
        columns={"size": "total_passes_for_lanes"}
    )
    
    # Pivot para tener ext e int en columnas separadas
    pivot_obv = obv_by_group.pivot_table(
        index="pid",
        columns="start_group",
        values="obv_total",
        fill_value=0.0
    ).reset_index()
    
    pivot_obv.columns.name = None
    
    # Asegurar que existen ambas columnas
    if "ext" not in pivot_obv.columns:
        pivot_obv["ext"] = 0.0
    if "int" not in pivot_obv.columns:
        pivot_obv["int"] = 0.0
    
    pivot_obv = pivot_obv.rename(columns={
        "ext": "obv_from_ext_total",
        "int": "obv_from_int_total",
        "pid": "player_id"
    })
    
    # === PASO 5: Merge con minutos + calcular per90 ===
    result = pivot_obv.merge(
        player_minutes_summary[["player_id", "total_minutes"]],
        on="player_id",
        how="inner"
    )
    
    # Agregar conteo de pases (renombrar pid a player_id primero)
    passes_by_player = passes_by_player.rename(columns={"pid": "player_id"})
    result = result.merge(passes_by_player, on="player_id", how="left")
    result["total_passes_for_lanes"] = result["total_passes_for_lanes"].fillna(0)
    
    # Normalizar per90
    m = result["total_minutes"].replace({0: np.nan})
    result["obv_from_ext_per90"] = result["obv_from_ext_total"] / m * 90.0
    result["obv_from_int_per90"] = result["obv_from_int_total"] / m * 90.0
    
    # === PASO 6: Calcular índice de sesgo ===
    # Solo calcular si tiene suficientes pases
    result["lane_bias_index"] = np.nan
    
    mask = result["total_passes_for_lanes"] >= min_passes
    
    if mask.any():
        eps = 1e-9
        den = (result.loc[mask, "obv_from_int_per90"].fillna(0).abs() + 
               result.loc[mask, "obv_from_ext_per90"].fillna(0).abs()) + eps
        
        result.loc[mask, "lane_bias_index"] = (
            (result.loc[mask, "obv_from_int_per90"].fillna(0) - 
             result.loc[mask, "obv_from_ext_per90"].fillna(0)) / den
        )
    
    players_with_bias = mask.sum()
    print(f"  ✓ Jugadores con ≥{min_passes} pases: {players_with_bias:,}")
    
    # Limpiar columnas para output
    result = result[[
        "player_id",
        "obv_from_ext_per90",
        "obv_from_int_per90",
        "lane_bias_index",
        "total_passes_for_lanes"
    ]]
    
    return result