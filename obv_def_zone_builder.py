# -*- coding: utf-8 -*-
"""
cb_zone_builder.py
Calcula métricas de zona de influencia defensiva para todos los jugadores.

Para cada jugador en cada partido:
  1. Toma sus acciones defensivas → construye un rectángulo PCA (zona)
  2. Invierte coordenadas del rival (StatsBomb muestra cada equipo atacando hacia x=120)
  3. Mide cuánto daño genera el rival dentro y desde esa zona

Métricas retornadas (per90):
  - obv_into_area_per90:        OBV rival generado DENTRO de la zona
  - obv_from_area_per90:        OBV rival generado DESPUÉS de entrar a la zona (por posesión)
  - shots_from_area_per90:      Tiros rivales surgidos de posesiones que tocaron la zona
  - xg_from_area_per90:         xG de esos tiros

IMPORTANTE:
  - Se llama DESDE main_analysis.py
  - Reutiliza df_all (con x, y, _player_name, type, team, possession, obv_total_net)
  - Reutiliza player_minutes_summary para normalizar per90
  - No calcula minutos ni filtra por posición (lo hace main/scoring)
"""

import numpy as np
import pandas as pd
from matplotlib.path import Path as MplPath


# ============= TIPOS DE ACCIONES DEFENSIVAS =============
DEFENSIVE_TYPES = {
    "Pressure", "Ball Recovery", "Duel", "Clearance", "Block",
    "Interception", "Foul Won", "Shield", "50/50", "Dispossessed", "Dribbled Past"
}


# ============= HELPERS GEOMÉTRICOS (únicos de este módulo) =============

def _pca_rect(x, y, q=0.15, scale=1.0):
    """
    Construye un rectángulo PCA orientado óptimamente alrededor de las acciones defensivas.
    
    PCA encuentra la orientación que mejor explica la dispersión de puntos.
    El rectángulo se recorta en los cuantiles q y (1-q) para ignorar outliers.
    
    Retorna array (4,2) con los 4 vértices del rectángulo rotado, o None si no hay suficientes puntos.
    """
    X = np.column_stack([x, y]).astype(float)
    X = X[~np.isnan(X).any(axis=1)]
    if len(X) < 5:
        return None

    mu = X.mean(axis=0)
    C = np.cov((X - mu).T)
    eigvals, eigvecs = np.linalg.eigh(C)
    V = eigvecs[:, np.argsort(eigvals)[::-1]]  # ordenar por varianza descendente

    # Proyectar al espacio PCA
    Y = (X - mu) @ V

    # Recortar outliers con cuantiles
    x1, x2 = np.nanquantile(Y[:, 0], [q, 1 - q])
    y1, y2 = np.nanquantile(Y[:, 1], [q, 1 - q])

    # Escalar alrededor del centro
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    x1, x2 = cx + (x1 - cx) * scale, cx + (x2 - cx) * scale
    y1, y2 = cy + (y1 - cy) * scale, cy + (y2 - cy) * scale

    # Rectángulo en espacio PCA → proyectar de vuelta al espacio original
    rect_pca = np.array([[x1, y1], [x2, y1], [x2, y2], [x1, y2]])
    return rect_pca @ V.T + mu


def _inside_poly(x, y, poly):
    """Boolean mask: qué puntos caen dentro del polígono."""
    if poly is None:
        return np.zeros(len(x), dtype=bool)
    path = MplPath(np.vstack([poly, poly[0]]))
    return path.contains_points(np.column_stack([x, y]), radius=0.01)


def _time_key(df):
    """
    Clave temporal para ordenar eventos dentro de una posesión.
    Necesaria para determinar cuál fue el PRIMER contacto del rival en la zona.
    """
    if "timestamp" in df.columns:
        t = pd.to_timedelta(df["timestamp"].astype(str), errors="coerce").dt.total_seconds()
        if t.notna().any():
            return t.fillna(0)
    per = pd.to_numeric(df.get("period", 1), errors="coerce").fillna(1)
    minute = pd.to_numeric(df.get("minute", 0), errors="coerce").fillna(0)
    second = pd.to_numeric(df.get("second", 0), errors="coerce").fillna(0)
    return (per - 1) * 3600 + minute * 60 + second


# ============= CÁLCULO POR PARTIDO =============

def _zone_metrics_one_match(df_match, player_name, df_rival, q, scale, min_def_actions):
    """
    Calcula métricas de zona para UN jugador en UN partido.

    df_rival: DataFrame rival ya preparado (coordenadas invertidas, columna _t calculada).
              Se comparte entre todos los jugadores del mismo equipo → no se modifica aquí.

    Retorna dict {obv_into, obv_from, shots_initiated, xg_from} o None si no hay datos.
    """
    # Eventos del jugador en este partido
    df_player = df_match[df_match["_player_name"] == player_name]
    if df_player.empty:
        return None

    # Acciones defensivas con coordenadas válidas
    df_def = df_player[df_player["type"].isin(DEFENSIVE_TYPES)].dropna(subset=["x", "y"])
    if len(df_def) < min_def_actions:
        return None

    # Construir zona PCA
    rect = _pca_rect(df_def["x"].values, df_def["y"].values, q=q, scale=scale)
    if rect is None:
        return None

    # --- OBV dentro de la zona ---
    inside = _inside_poly(df_rival["x"].values, df_rival["y"].values, rect)
    obv_into = df_rival.loc[inside, "obv_total_net"].sum(skipna=True)

    # --- OBV downstream por posesión (vectorizado) ---
    # Lógica: dentro de cada posesión rival, desde el momento que el primer evento
    # cae dentro de la zona, todo el OBV posterior en esa posesión cuenta.
    #
    # 1) t0_per_poss: por cada posesión que tocó la zona, el min de _t donde inside==True
    # 2) mapped_t0: ese t0 mapeado a cada fila (NaN si la posesión nunca tocó la zona)
    # 3) down: True donde la posesión tocó la zona Y _t >= t0
    t0_per_poss = (df_rival.loc[inside, ["possession", "_t"]]
                   .groupby("possession")["_t"]
                   .min())

    mapped_t0 = df_rival["possession"].map(t0_per_poss)
    down = mapped_t0.notna() & (df_rival["_t"] >= mapped_t0)

    obv_from = df_rival.loc[down, "obv_total_net"].sum(skipna=True)

    # --- Tiros surgidos desde la zona ---
    # Un tiro surge de la zona si cumple la misma condición `down`: su posesión
    # tocó la zona y el tiro ocurre en o después del primer contacto.
    shots_from_zone = df_rival.loc[down & (df_rival["type"] == "Shot")]

    xg_col = "shot_statsbomb_xg" if "shot_statsbomb_xg" in df_rival.columns else None
    xg_from = 0.0
    if xg_col and not shots_from_zone.empty:
        xg_from = pd.to_numeric(shots_from_zone[xg_col], errors="coerce").sum(skipna=True)
        if np.isnan(xg_from):
            xg_from = 0.0

    return {
        "obv_into": float(obv_into),
        "obv_from": float(obv_from),
        "shots_initiated": int(len(shots_from_zone)),
        "xg_from": float(xg_from),
    }


# ============= FUNCIÓN PRINCIPAL =============

def calculate_cb_zone_metrics(
    df,
    player_minutes_summary,
    pitch_length=120.0,
    pitch_width=80.0,
    q=0.15,
    scale=1.0,
    min_def_actions=5,
):
    """
    Calcula métricas de zona de influencia defensiva para todos los jugadores.
    
    Args:
        df: DataFrame de eventos (df_all de main_analysis).
            Necesita: _player_name, team, type, x, y, possession, obv_total_net, match_id
        player_minutes_summary: DataFrame de main_analysis.
            Necesita: player_id, player_name, total_minutes
        pitch_length: Largo de cancha (default 120)
        pitch_width: Ancho de cancha (default 80)
        q: Cuantil PCA para recorte de outliers (default 0.15)
        scale: Escala del rectángulo PCA (default 1.0)
        min_def_actions: Mínimo de acciones defensivas por partido para calcular zona (default 5)
        
    Returns:
        DataFrame con columnas:
        - player_id
        - obv_into_area_per90
        - obv_from_area_per90
        - shots_from_area_per90
        - xg_from_area_per90
    """
    print("\n--- Calculando métricas de zona defensiva ---")

    # Verificar columnas necesarias
    required = ["_player_name", "team", "type", "x", "y", "possession", "obv_total_net", "match_id"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        print(f"  ⚠️  Columnas faltantes en df: {missing}")
        return pd.DataFrame()

    # Normalizar tipo a string
    df = df.copy()
    df["type"] = df["type"].astype(str).str.strip()

    # Pre-agrupar por partido (carga cada match una sola vez)
    match_groups = {mid: g for mid, g in df.groupby("match_id")}
    print(f"  Partidos a procesar: {len(match_groups):,}")

    # Lookup: player_name → {player_id, total_minutes}
    player_lookup = {}
    for _, row in player_minutes_summary.iterrows():
        player_lookup[row["player_name"]] = {
            "player_id": row["player_id"],
            "total_minutes": row["total_minutes"],
        }

    # Acumulador por jugador
    player_totals = {}  # player_name → {obv_into, obv_from, shots, xg}

    total_matches = len(match_groups)
    for i, (mid, df_match) in enumerate(match_groups.items()):
        if (i + 1) % 100 == 0:
            print(f"    Procesando partido {i+1}/{total_matches}...")

        # --- Pre-computar rival por equipo (una sola vez por partido) ---
        # En un partido hay 2 equipos. El rival de cada equipo es el otro.
        # Esto se hacía antes dentro de _zone_metrics_one_match (por cada jugador),
        # generando la misma copia invertida 11 veces. Ahora se hace 2 veces.
        teams = df_match["team"].dropna().unique()
        rivals_by_team = {}
        for team in teams:
            df_rival = df_match[df_match["team"] != team].dropna(subset=["x", "y"]).copy()
            if df_rival.empty:
                continue
            df_rival["x"] = pitch_length - df_rival["x"]
            df_rival["y"] = pitch_width - df_rival["y"]
            df_rival["_t"] = _time_key(df_rival)
            rivals_by_team[team] = df_rival

        # Mapeo jugador → equipo en este partido (un solo groupby)
        player_team = df_match.dropna(subset=["_player_name"]).groupby("_player_name")["team"].first().to_dict()

        # Solo jugadores que existan en player_minutes_summary
        players_in_match = df_match["_player_name"].dropna().unique()

        for pname in players_in_match:
            if pname not in player_lookup:
                continue

            team_player = player_team.get(pname)
            if team_player not in rivals_by_team:
                continue

            m = _zone_metrics_one_match(
                df_match, pname, rivals_by_team[team_player],
                q=q, scale=scale,
                min_def_actions=min_def_actions,
            )

            if m is None:
                continue

            # Acumular
            if pname not in player_totals:
                player_totals[pname] = {"obv_into": 0.0, "obv_from": 0.0, "shots": 0, "xg": 0.0}

            player_totals[pname]["obv_into"] += m["obv_into"]
            player_totals[pname]["obv_from"] += m["obv_from"]
            player_totals[pname]["shots"] += m["shots_initiated"]
            player_totals[pname]["xg"] += m["xg_from"]

    if not player_totals:
        print("  ⚠️  No se generaron métricas de zona")
        return pd.DataFrame()

    # Construir DataFrame de resultado con per90
    results = []
    for pname, totals in player_totals.items():
        info = player_lookup[pname]
        total_minutes = info["total_minutes"]
        if total_minutes <= 0:
            continue

        per90 = 90.0 / total_minutes
        results.append({
            "player_id": info["player_id"],
            "obv_into_area_per90": round(totals["obv_into"] * per90, 4),
            "obv_from_area_per90": round(totals["obv_from"] * per90, 4),
            "shots_from_area_per90": round(totals["shots"] * per90, 4),
            "xg_from_area_per90": round(totals["xg"] * per90, 4),
        })

    result_df = pd.DataFrame(results)
    print(f"  ✓ Jugadores con métricas de zona: {len(result_df):,}")

    return result_df