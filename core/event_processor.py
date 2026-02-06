# -*- coding: utf-8 -*-
# core/event_processor.py

import sys
from pathlib import Path
import pandas as pd
import numpy as np
import json, ast
import sys
from datetime import datetime

# Agregar directorio raíz al Python path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# Importar desde config centralizado
from config import (
    BASE_DIR,
    DATA_DIR,
    OUTPUTS_DIR,
    EVENTS_CSV,
    Processing,
    ensure_directories
)

# Importar funciones de turnover
from turnover_calculator import compute_player_turnovers

# Importar módulo de análisis de carriles OBV
try:
    from obv_lanes import calculate_lane_bias_metrics
    OBV_LANES_AVAILABLE = True
    print("✓ Módulo obv_lanes importado correctamente")
except ImportError as e:
    print(f"⚠️  Módulo obv_lanes no disponible: {e}")
    print("   Asegúrate de que obv_lanes.py está en el mismo directorio")
    OBV_LANES_AVAILABLE = False
except Exception as e:
    print(f"❌ Error importando obv_lanes: {e}")
    OBV_LANES_AVAILABLE = False

# Importar módulo de análisis de zonas defensivas OBV
try:
    from obv_zones import calculate_cb_zone_metrics
    CB_ZONE_AVAILABLE = True
    print("✓ Módulo cb_zone_builder importado correctamente")
except ImportError as e:
    print(f"⚠️  Módulo cb_zone_builder no disponible: {e}")
    CB_ZONE_AVAILABLE = False
except Exception as e:
    print(f"❌ Error importando cb_zone_builder: {e}")
    CB_ZONE_AVAILABLE = False

# Importar módulo de métricas de arqueros
try:
    from goalkeeper_metrics import calculate_gk_metrics
    GK_METRICS_AVAILABLE = True
    print("✓ Módulo goalkeeper_metrics importado correctamente")
except ImportError as e:
    print(f"⚠️  Módulo goalkeeper_metrics no disponible: {e}")
    print("   Asegúrate de que goalkeeper_metrics.py está en el mismo directorio")
    GK_METRICS_AVAILABLE = False
except Exception as e:
    print(f"❌ Error importando goalkeeper_metrics: {e}")
    GK_METRICS_AVAILABLE = False

# ============= CONFIG =============
# Path del archivo de eventos desde config
PATH = str(EVENTS_CSV)
season = EVENTS_CSV.name.split('_', 1)[1].replace('.csv', '')  # ← Cambiar esta línea

# Configuraciones desde config.Processing
ASSUME_END_CAP = Processing.ASSUME_END_CAP
OUTPUT_DIR = str(OUTPUTS_DIR)

# ============= LOGGING SETUP =============
class Logger:
    """Clase para escribir simultáneamente en consola y archivo"""
    def __init__(self, filename):
        self.terminal = sys.stdout
        
        # Asegurar que el directorio padre existe
        filepath = Path(filename)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        self.log = open(filename, 'w', encoding='utf-8')
        
    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
        
    def flush(self):
        self.terminal.flush()
        self.log.flush()
        
    def close(self):
        self.log.close()

# Asegurar que directorio de outputs existe
ensure_directories()

# Configurar logging
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
log_filename = OUTPUTS_DIR / f"analysis_log_{season}_{timestamp}.txt"
sys.stdout = Logger(str(log_filename))

print("="*70)
print(f"ANÁLISIS DE EVENTOS - TEMPORADA {season}")
print(f"Inicio: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print("="*70)
print(f"\n📁 Configuración de paths:")
print(f"  Events CSV: {EVENTS_CSV}")
print(f"  Output Dir: {OUTPUTS_DIR}")
print(f"\n⚙️  Configuración de procesamiento:")
print(f"  ASSUME_END_CAP: {ASSUME_END_CAP}")

# ============= COLUMNAS A DISCRIMINAR =============
COLUMNS_TO_DISCRIMINATE = [
    "type",
    "pass_height",
    "pass_type",
    "pass_switch",
    "duel_type",
    "shot_type",
    "play_pattern",
]

# ============= COLUMNAS SOLO PARA CONTAR EVENTOS (sin métricas) =============
COLUMNS_TO_COUNT_ONLY = [
    "duel_outcome",
    "interception_outcome",
    "pass_outcome",  # ← NUEVO: para calcular % de pases completados
]

# ============= CONFIG TERCIOS DE CANCHA =============
ENABLE_THIRDS_ANALYSIS = Processing.ENABLE_THIRDS_ANALYSIS

THIRDS_EVENTS = [
    "Carry",
    "Pass",
    "Ball Recovery",
    "Duel",
    "Interception",
    "Pressure",
    "Dispossessed"
]

ENABLE_CROSS_ATTACKING = Processing.ENABLE_CROSS_ATTACKING

# ============= CONFIG TURNOVERS =============
ENABLE_TURNOVER_ANALYSIS = Processing.ENABLE_TURNOVER_ANALYSIS
TURNOVER_OPEN_PLAY_ONLY = Processing.TURNOVER_OPEN_PLAY_ONLY
TURNOVER_EXCLUDE_RESTARTS = Processing.TURNOVER_EXCLUDE_RESTARTS

# ============= MÉTRICAS A DISCRIMINAR =============
METRICS_TO_SPLIT = ["obv_total_net", "shot_statsbomb_xg"]
METRICS_FOR_THIRDS = ["obv_total_net"]

# ============= HELPERS =============
def _coerce_literal(x):
    if isinstance(x, str):
        s = x.strip()
        if s.startswith(("{", "[")):
            try:
                return json.loads(s)
            except Exception:
                try:
                    return ast.literal_eval(s)
                except Exception:
                    return x
    return x

def extract_x_y_from_location(location_val):
    """
    Extrae coordenadas (x, y) desde columna 'location'.
    Formato esperado: [61.0, 40.1] o "[61.0, 40.1]"
    Retorna: (x, y) o (None, None)
    """
    if pd.isna(location_val):
        return None, None
    
    # Parsear si es string
    loc = _coerce_literal(location_val)
    
    # Debe ser lista/tupla con al menos 2 elementos
    if isinstance(loc, (list, tuple)) and len(loc) >= 2:
        try:
            return float(loc[0]), float(loc[1])
        except (ValueError, TypeError):
            return None, None
    
    return None, None

def get_value_from_column(val):
    """Extrae valor de columnas que pueden ser dict o string."""
    if pd.isna(val):
        return None
    v = _coerce_literal(val)
    if isinstance(v, dict):
        return v.get("name") or v.get("label") or v.get("type")
    return str(v)

def get_player_id_series(df):
    """player_id desde columna plana o dentro de 'player' (dict)."""
    for cand in ["player_id", "playerId", "idPlayer"]:
        if cand in df.columns:
            return pd.to_numeric(df[cand], errors="coerce")
    if "player" in df.columns:
        obj = df["player"].apply(_coerce_literal)
        return obj.apply(lambda v: v.get("id") if isinstance(v, dict) else np.nan)
    return pd.Series(np.nan, index=df.index)

def get_player_name_series(df):
    """Nombre desde 'player' (dict) o variantes planas."""
    if "player" in df.columns:
        obj = df["player"].apply(_coerce_literal)
        return obj.apply(lambda v: v.get("name") if isinstance(v, dict) else str(v))
    for cand in ["player.name", "player_name", "playerName"]:
        if cand in df.columns:
            return df[cand].astype(str)
    return pd.Series("", index=df.index)

def get_position_str(val):
    """Devuelve position.name como string plano (si está)."""
    if pd.isna(val):
        return None
    v = _coerce_literal(val)
    if isinstance(v, dict):
        return v.get("name") or v.get("label")
    return str(v)

def get_team_name(val):
    """Extrae nombre del equipo desde dict o string."""
    if pd.isna(val):
        return None
    v = _coerce_literal(val)
    if isinstance(v, dict):
        return v.get("name") or v.get("team_name")
    return str(val)

def match_end_minute(df_match):
    """Final del partido por minuto máximo observado (cap a ASSUME_END_CAP; mínimo 90)."""
    mx = pd.to_numeric(df_match.get("minute", pd.Series([], dtype=float)), errors="coerce").max()
    if pd.isna(mx):
        return 90.0
    return float(min(ASSUME_END_CAP, max(90.0, mx)))

def lineup_players_from_match(df_match):
    """
    Extrae XI titular del partido desde eventos type='Starting XI' -> tactics.lineup.
    Devuelve dict: {player_id: {'position': pos_name, 'team': team_name}}
    """
    out = {}
    xi = df_match.loc[df_match["type"].astype(str) == "Starting XI"]
    if xi.empty or "tactics" not in xi.columns:
        return out
    
    for idx, row in xi.iterrows():
        tac_obj = _coerce_literal(row["tactics"])
        team_val = get_team_name(row.get("team")) if "team" in row else None
        
        if isinstance(tac_obj, dict):
            for p in tac_obj.get("lineup", []):
                pid = (p.get("player") or {}).get("id")
                pos = (p.get("position") or {}).get("name") or p.get("position")
                if pid is not None:
                    out[int(pid)] = {
                        'position': str(pos) if pos else None,
                        'team': team_val
                    }
    return out

def sub_on_minute(df_match, player_id):
    """Minuto en que ENTRA: Substitution con substitution_replacement = jugador."""
    subs = df_match.loc[df_match["type"].astype(str) == "Substitution"]
    if subs.empty or "substitution_replacement" not in subs.columns:
        return None
    mins = []
    for _, row in subs.iterrows():
        repl = _coerce_literal(row["substitution_replacement"])
        if isinstance(repl, dict) and repl.get("id") == player_id:
            m = pd.to_numeric(row.get("minute", np.nan), errors="coerce")
            if pd.notna(m):
                mins.append(float(m))
    return min(mins) if mins else None

def sub_off_minute(df_match, player_id):
    """Minuto en que SALE: Substitution donde 'player' es el que sale."""
    subs = df_match.loc[df_match["type"].astype(str) == "Substitution"]
    if subs.empty:
        return None
    ids = get_player_id_series(subs)
    mask = (ids == player_id)
    if mask.any():
        mins = pd.to_numeric(subs.loc[mask, "minute"], errors="coerce").dropna()
        if not mins.empty:
            return float(mins.min())
    return None

def minutes_played_in_match(df_match, player_id, xi_players):
    """
    Minutos jugados por player_id en match:
      - Si está en XI → desde 0 hasta sub_off (si existe) o fin del partido.
      - Si no está en XI → si entra por sub_on → desde on hasta sub_off o fin.
      - Si no hay señales → 0.
    """
    end_m = match_end_minute(df_match)
    started = player_id in xi_players
    on = sub_on_minute(df_match, player_id)
    off = sub_off_minute(df_match, player_id)

    if started:
        if off is not None:
            return int(round(max(0.0, off)))
        return int(round(end_m))
    else:
        if on is not None and off is None:
            return int(round(max(0.0, end_m - on)))
        if on is not None and off is not None:
            return int(round(max(0.0, off - on)))
    return 0

def infer_position_from_events(df_match, player_id):
    """
    Infiere la posición más frecuente de un jugador en sus eventos del partido.
    Útil para suplentes que no están en el XI.
    """
    ids = get_player_id_series(df_match)
    player_events = df_match.loc[ids == player_id]
    
    if player_events.empty or "position" not in player_events.columns:
        return None
    
    pos_vals = player_events["position"].apply(get_position_str).dropna()
    if pos_vals.empty:
        return None
    
    return pos_vals.value_counts().idxmax()

# ============= HELPERS PARA TERCIOS =============
def infer_pitch_dimensions(df: pd.DataFrame) -> tuple:
    """
    Infiere dimensiones de la cancha (length, width) desde los datos.
    StatsBomb usa 120x80 por defecto.
    MODIFICADO: Ahora usa la columna 'location' en lugar de 'x' e 'y'.
    """
    # Extraer coordenadas desde 'location'
    if "location" not in df.columns:
        print("⚠️ Columna 'location' no encontrada, usando dimensiones por defecto")
        return 120.0, 80.0
    
    coords = df["location"].apply(extract_x_y_from_location)
    x_vals = coords.apply(lambda c: c[0])
    y_vals = coords.apply(lambda c: c[1])
    
    # Ancho (Y)
    y = pd.to_numeric(y_vals, errors="coerce")
    if not y.notna().any():
        width = 80.0
    else:
        ymax = y.dropna().quantile(0.995)
        if ymax <= 82:
            width = 80.0
        elif ymax <= 102:
            width = 100.0
        else:
            width = float(min(120.0, max(80.0, y.dropna().max())))
    
    # Largo (X)
    x = pd.to_numeric(x_vals, errors="coerce")
    if not x.notna().any():
        length = 120.0
    else:
        xmax = x.dropna().quantile(0.995)
        length = float(min(120.0, max(100.0, xmax)))
    
    return length, width

def get_third_from_x(x, length=120.0):
    """
    Determina el tercio de la cancha según coordenada X.
    - defensive: 0 - length/3
    - middle: length/3 - 2*length/3
    - attacking: 2*length/3 - length
    """
    if pd.isna(x):
        return None
    x = float(x)
    third_boundary = length / 3.0
    
    if x < third_boundary:
        return "defensive"
    elif x < 2 * third_boundary:
        return "middle"
    else:
        return "attacking"

# ============= LECTURA =============
print("Cargando datos...")
df = pd.read_csv(PATH, low_memory=False)

df["season"] = season

df["minute"] = pd.to_numeric(df.get("minute", np.nan), errors="coerce")
df["type"] = df["type"].astype(str)

df["_player_id"] = get_player_id_series(df)
df["_player_name"] = get_player_name_series(df)

# ============= EXTRAER COORDENADAS X, Y DESDE 'location' =============
print("\n📍 Extrayendo coordenadas desde columna 'location'...")

if "location" in df.columns:
    # Extraer x e y desde location
    coords = df["location"].apply(extract_x_y_from_location)
    df["x"] = coords.apply(lambda c: c[0])
    df["y"] = coords.apply(lambda c: c[1])
    
    # Estadísticas
    x_count = df["x"].notna().sum()
    y_count = df["y"].notna().sum()
    print(f"  ✓ Coordenadas X extraídas: {x_count:,} eventos ({x_count/len(df)*100:.1f}%)")
    print(f"  ✓ Coordenadas Y extraídas: {y_count:,} eventos ({y_count/len(df)*100:.1f}%)")
else:
    print("  ⚠️ Columna 'location' no encontrada")
    # Verificar si ya existen x, y
    if "x" in df.columns and "y" in df.columns:
        print("  ✓ Usando columnas 'x' e 'y' existentes")
    else:
        print("  ❌ ERROR: No se encontraron ni 'location' ni columnas 'x', 'y'")
        print("  El análisis por tercios no estará disponible")

# ============= EXTRAER COORDENADAS FINALES (PASS Y CARRY) =============
print("\n📍 Extrayendo coordenadas finales (pass_end_location, carry_end_location)...")

# Pass end location
if "pass_end_location" in df.columns:
    coords_pass_end = df["pass_end_location"].apply(extract_x_y_from_location)
    df["x_pass_end"] = coords_pass_end.apply(lambda c: c[0])
    df["y_pass_end"] = coords_pass_end.apply(lambda c: c[1])
    
    pass_end_count = df["x_pass_end"].notna().sum()
    print(f"  ✓ Pass end X extraídas: {pass_end_count:,} eventos ({pass_end_count/len(df)*100:.1f}%)")
else:
    print("  ⚠️ Columna 'pass_end_location' no encontrada")
    df["x_pass_end"] = None
    df["y_pass_end"] = None

# Carry end location
if "carry_end_location" in df.columns:
    coords_carry_end = df["carry_end_location"].apply(extract_x_y_from_location)
    df["x_carry_end"] = coords_carry_end.apply(lambda c: c[0])
    df["y_carry_end"] = coords_carry_end.apply(lambda c: c[1])
    
    carry_end_count = df["x_carry_end"].notna().sum()
    print(f"  ✓ Carry end X extraídas: {carry_end_count:,} eventos ({carry_end_count/len(df)*100:.1f}%)")
else:
    print("  ⚠️ Columna 'carry_end_location' no encontrada")
    df["x_carry_end"] = None
    df["y_carry_end"] = None

print(f"\nTotal de eventos cargados: {len(df):,}")

# ============= 1) IDENTIFICAR JUGADORES =============
print("\nIdentificando jugadores...")

player_info = {}

for mid, g in df.groupby("match_id", sort=False):
    xi_players = lineup_players_from_match(g)
    for pid, info in xi_players.items():
        if pid not in player_info:
            player_info[pid] = {'name': None, 'positions': set(), 'teams': set()}
        if info.get('position'):
            player_info[pid]['positions'].add(info['position'])
        if info.get('team'):
            player_info[pid]['teams'].add(info['team'])

if "position" in df.columns:
    pos_events = df[["_player_id", "_player_name", "position"]].dropna(subset=["_player_id"])
    for _, row in pos_events.iterrows():
        pid = int(row["_player_id"])
        if pid not in player_info:
            player_info[pid] = {'name': None, 'positions': set(), 'teams': set()}
        player_info[pid]['name'] = row["_player_name"]
        pos = get_position_str(row["position"])
        if pos:
            player_info[pid]['positions'].add(pos)

name_events = df[["_player_id", "_player_name"]].dropna()
for _, row in name_events.iterrows():
    pid = int(row["_player_id"])
    if pid in player_info and not player_info[pid]['name']:
        player_info[pid]['name'] = row["_player_name"]

if "team" in df.columns:
    team_events = df[["_player_id", "team"]].dropna(subset=["_player_id"])
    for _, row in team_events.iterrows():
        pid = int(row["_player_id"])
        if pid in player_info:
            team = get_team_name(row["team"])
            if team:
                player_info[pid]['teams'].add(team)

print(f"Jugadores únicos identificados: {len(player_info):,}")

# ============= 1B) ANALIZAR TURNOVERS =============
if ENABLE_TURNOVER_ANALYSIS:
    print("\n" + "="*70)
    print("ANALIZANDO TURNOVERS")
    print("="*70)
    
    try:
        # Preparar DataFrame para turnovers
        # Necesitamos asegurar que team y possession_team estén como strings limpias
        df_turnovers = df.copy()
        
        if "team" in df_turnovers.columns:
            df_turnovers["team"] = df_turnovers["team"].apply(get_team_name)
        
        if "possession_team" in df_turnovers.columns:
            df_turnovers["possession_team"] = df_turnovers["possession_team"].apply(get_team_name)
        
        # Asegurar que player_id esté disponible
        if "player_id" not in df_turnovers.columns:
            df_turnovers["player_id"] = df_turnovers["_player_id"]
        
        print(f"Calculando turnovers con configuración:")
        print(f"  - Open play only: {TURNOVER_OPEN_PLAY_ONLY}")
        print(f"  - Exclude restarts: {TURNOVER_EXCLUDE_RESTARTS}")
        
        turnovers_df = compute_player_turnovers(
            df_turnovers,
            open_play_only=TURNOVER_OPEN_PLAY_ONLY,
            exclude_restart_patterns=TURNOVER_EXCLUDE_RESTARTS
        )
        
        print(f"\n✓ Turnovers detectados: {len(turnovers_df):,}")
        
        if not turnovers_df.empty:
            # Estadísticas generales
            print(f"\nEstadísticas de turnovers:")
            print(f"  - Jugadores únicos con turnovers: {turnovers_df['player_id'].nunique():,}")
            print(f"  - Partidos con turnovers: {turnovers_df['match_id'].nunique():,}")
            
            # Top tipos de turnovers
            if "turnover_how" in turnovers_df.columns:
                print(f"\nTop 10 tipos de turnovers:")
                print(turnovers_df["turnover_how"].value_counts().head(10))
            
            # Agregar turnovers por jugador
            turnover_counts = (
                turnovers_df
                .groupby("player_id")
                .size()
                .reset_index(name="total_turnovers")
            )
            
            # Agregar por tipo de turnover
            if "turnover_how" in turnovers_df.columns:
                turnover_by_type = (
                    turnovers_df
                    .groupby(["player_id", "turnover_how"])
                    .size()
                    .reset_index(name="count")
                )
                
                # Pivot para tener columnas por tipo
                turnover_pivot = turnover_by_type.pivot(
                    index="player_id",
                    columns="turnover_how",
                    values="count"
                ).fillna(0).reset_index()
                
                # Renombrar columnas
                turnover_pivot.columns = [
                    f"turnovers_{col.lower().replace(' ', '_').replace(':', '')}" if col != "player_id" else col
                    for col in turnover_pivot.columns
                ]
                
                # Merge con conteo total
                turnover_counts = turnover_counts.merge(turnover_pivot, on="player_id", how="left")
            
            print(f"\n✓ Métricas de turnovers agregadas por jugador")
            print(f"  - Columnas de turnovers generadas: {len([c for c in turnover_counts.columns if c.startswith('turnovers_')])}")
            
    except Exception as e:
        print(f"\n❌ Error al calcular turnovers: {e}")
        print("  Continuando sin análisis de turnovers...")
        ENABLE_TURNOVER_ANALYSIS = False

# ============= 2) CALCULAR MINUTOS POR PARTIDO CON POSICIÓN =============
print("\nCalculando minutos jugados por partido (con tracking de posición)...")

minutes_rows = []
match_count = 0

for mid, g in df.groupby("match_id", sort=False):
    match_count += 1
    if match_count % 100 == 0:
        print(f"  Procesando partido {match_count}...")
    
    xi_players = lineup_players_from_match(g)
    season_match = g["season"].iloc[0] if "season" in g.columns else None
    
    participants = set(xi_players.keys())
    
    subs = g.loc[g["type"] == "Substitution"]
    if not subs.empty:
        ids_out = get_player_id_series(subs).dropna().astype(int).tolist()
        participants.update(ids_out)
        if "substitution_replacement" in subs.columns:
            for repl in subs["substitution_replacement"].dropna().tolist():
                obj = _coerce_literal(repl)
                if isinstance(obj, dict) and obj.get("id") is not None:
                    participants.add(int(obj["id"]))
    
    participants.update(get_player_id_series(g).dropna().astype(int).tolist())
    
    for pid in participants:
        mins = minutes_played_in_match(g, int(pid), xi_players)
        if mins > 0:
            is_starter = pid in xi_players
            
            if pid in xi_players:
                position_match = xi_players[pid]['position']
                team_match = xi_players[pid]['team']
            else:
                position_match = infer_position_from_events(g, int(pid))
                team_match = None
                if "team" in g.columns:
                    player_events = g.loc[get_player_id_series(g) == pid]
                    if not player_events.empty:
                        team_match = get_team_name(player_events["team"].iloc[0])
            
            minutes_rows.append({
                "player_id": int(pid),
                "match_id": mid,
                "season": season_match,
                "minutes": int(mins),
                "starter": is_starter,
                "position_match": position_match,
                "team": team_match
            })

minutes_df = pd.DataFrame(minutes_rows)
print(f"Total de apariciones jugador-partido: {len(minutes_df):,}")

# ============= 3) RESUMEN DE MINUTOS CON POSICIONES =============
print("\nGenerando resumen de minutos (con análisis de posiciones)...")

player_minutes_summary = minutes_df.groupby("player_id").agg({
    "minutes": "sum",
    "match_id": "nunique",
    "starter": "sum",
    "season": lambda x: list(x.unique())
}).reset_index()

player_minutes_summary.columns = ["player_id", "total_minutes", "matches_played", "times_starter", "seasons"]

position_minutes = (
    minutes_df
    .dropna(subset=["position_match"])
    .groupby(["player_id", "position_match"], as_index=False)["minutes"]
    .sum()
    .rename(columns={"minutes": "minutes_in_position"})
)

primary_position = (
    position_minutes
    .sort_values("minutes_in_position", ascending=False)
    .groupby("player_id")
    .first()
    .reset_index()[["player_id", "position_match", "minutes_in_position"]]
    .rename(columns={
        "position_match": "primary_position",
        "minutes_in_position": "minutes_primary_position"
    })
)

player_minutes_summary = player_minutes_summary.merge(primary_position, on="player_id", how="left")

player_minutes_summary["primary_position_share"] = (
    player_minutes_summary["minutes_primary_position"] / player_minutes_summary["total_minutes"]
)

player_minutes_summary["player_name"] = player_minutes_summary["player_id"].map(
    lambda pid: player_info.get(pid, {}).get('name', f"player_{pid}")
)
player_minutes_summary["all_positions"] = player_minutes_summary["player_id"].map(
    lambda pid: ", ".join(sorted(player_info.get(pid, {}).get('positions', set()))) or None
)
player_minutes_summary["teams"] = player_minutes_summary["player_id"].map(
    lambda pid: ", ".join(sorted(player_info.get(pid, {}).get('teams', set()))) or None
)

player_minutes_summary = player_minutes_summary[[
    "player_id", "player_name", "primary_position", "all_positions", "teams",
    "total_minutes", "minutes_primary_position", "primary_position_share",
    "matches_played", "times_starter", "seasons"
]]

print(f"\nEstadísticas de posiciones:")
print(f"  Jugadores con posición principal identificada: {player_minutes_summary['primary_position'].notna().sum()}")
print(f"  Posición principal promedio (% minutos): {player_minutes_summary['primary_position_share'].mean():.1%}")

if player_minutes_summary["primary_position"].notna().any():
    print(f"\nDistribución de posiciones principales (top 10):")
    print(player_minutes_summary["primary_position"].value_counts().head(10))

# ============= 4) CONVERTIR BOOLEANS Y PREPARAR MÉTRICAS =============
print("\nPreparando métricas numéricas...")

bool_cols = []
for c in df.columns:
    vals = df[c].dropna().astype(str).str.lower()
    if len(vals) == 0:
        continue
    if vals.isin(["true", "false"]).all():
        bool_cols.append(c)

print(f"Columnas booleanas detectadas: {len(bool_cols)}")

for c in bool_cols:
    df[c] = df[c].astype(str).str.lower().map({"true": 1, "false": 0}).astype("float64")

num_cols = df.select_dtypes(include=[np.number]).columns.tolist()

drop_cols = {
    "_player_id", "player_id", "team_id", "match_id", "possession_team_id",
    "possession", "period", "second", "index", "id", "minute", "x", "y", "z"
}
num_cols = [c for c in num_cols if c not in drop_cols]

print(f"Métricas numéricas a calcular: {len(num_cols)}")

# ============= 5) PREPARAR PARA DISCRIMINACIÓN =============
df["pid"] = get_player_id_series(df).astype("Int64")
all_player_ids = set(player_info.keys())
df_all = df[df["pid"].isin(all_player_ids)].copy()

if METRICS_TO_SPLIT is None:
    metrics_to_discriminate = num_cols.copy()
else:
    metrics_to_discriminate = [c for c in METRICS_TO_SPLIT if c in num_cols]

print(f"\nMétricas a discriminar (columnas): {metrics_to_discriminate}")

# ============= 5-SPECIAL) CALCULAR PASS/CARRY INTO FINAL THIRD =============
print("\n" + "="*70)
print("CALCULANDO PASS Y CARRY INTO FINAL THIRD")
print("="*70)

into_final_third_metrics = []

# Inferir dimensiones de la cancha
pitch_length, pitch_width = infer_pitch_dimensions(df_all)
print(f"Dimensiones de cancha: {pitch_length}m x {pitch_width}m")
attacking_third_threshold = (2 * pitch_length) / 3.0
print(f"Umbral tercio atacante: X >= {attacking_third_threshold:.1f}m")

# Pass into final third
if "x_pass_end" in df_all.columns and df_all["x_pass_end"].notna().any():
    df_all["pass_into_final_third"] = (
        (df_all["type"] == "Pass") & 
        (df_all["x_pass_end"] >= attacking_third_threshold)
    ).astype(int)
    
    pass_into_count = df_all["pass_into_final_third"].sum()
    players_with_pass_into = (df_all.groupby("pid")["pass_into_final_third"].sum() > 0).sum()
    
    print(f"✓ pass_into_final_third: {pass_into_count:,} eventos")
    print(f"  Jugadores con pases al tercio final: {players_with_pass_into:,}")
    
    into_final_third_metrics.append("pass_into_final_third")
    num_cols.append("pass_into_final_third")
else:
    print("⚠️ No se puede calcular pass_into_final_third (falta x_pass_end)")

# Carry into final third
if "x_carry_end" in df_all.columns and df_all["x_carry_end"].notna().any():
    df_all["carry_into_final_third"] = (
        (df_all["type"] == "Carry") & 
        (df_all["x_carry_end"] >= attacking_third_threshold)
    ).astype(int)
    
    carry_into_count = df_all["carry_into_final_third"].sum()
    players_with_carry_into = (df_all.groupby("pid")["carry_into_final_third"].sum() > 0).sum()
    
    print(f"✓ carry_into_final_third: {carry_into_count:,} eventos")
    print(f"  Jugadores con carries al tercio final: {players_with_carry_into:,}")
    
    into_final_third_metrics.append("carry_into_final_third")
    num_cols.append("carry_into_final_third")
else:
    print("⚠️ No se puede calcular carry_into_final_third (falta x_carry_end)")

if into_final_third_metrics:
    print(f"\n✓ Total métricas 'into final third' calculadas: {len(into_final_third_metrics)}")
else:
    print("\n⚠️ No se calcularon métricas 'into final third'")

# ============= 5A) DISCRIMINAR POR CADA COLUMNA =============
print("\n" + "="*70)
print("CREANDO MÉTRICAS DISCRIMINADAS POR COLUMNAS")
print("="*70)

all_discriminated_dfs = {}
discrimination_stats = []

for disc_column in COLUMNS_TO_DISCRIMINATE:
    print(f"\n--- Discriminando por: {disc_column} ---")
    
    if disc_column not in df_all.columns:
        print(f"  ⚠️ Columna '{disc_column}' no encontrada, saltando...")
        continue
    
    unique_values = df_all[disc_column].apply(get_value_from_column).dropna().unique()
    unique_values = [v for v in unique_values if v]
    
    print(f"  Valores únicos encontrados: {len(unique_values)}")
    
    for value in unique_values:
        df_value = df_all[
            df_all[disc_column].apply(get_value_from_column) == value
        ].copy()
        
        if len(df_value) == 0:
            continue
        
        value_sums = df_value.groupby("pid")[metrics_to_discriminate].sum(min_count=1).reset_index()
        value_sums = value_sums.rename(columns={"pid": "player_id"})
        
        suffix = f"{disc_column}_{str(value).lower().replace(' ', '_').replace('-', '_')}"
        
        rename_dict = {col: f"{col}_{suffix}" for col in metrics_to_discriminate}
        value_sums = value_sums.rename(columns=rename_dict)
        
        all_discriminated_dfs[suffix] = value_sums
        
        discrimination_stats.append({
            "discrimination_type": "column",
            "column": disc_column,
            "value": value,
            "suffix": suffix,
            "total_events": len(df_value),
            "unique_players": value_sums["player_id"].nunique()
        })
        
        print(f"  ✓ {value}: {len(df_value):,} eventos, {value_sums['player_id'].nunique()} jugadores")

# ============= 5A-COUNT) CONTAR EVENTOS SIN SUMAR MÉTRICAS =============
print("\n" + "="*70)
print("CONTANDO EVENTOS (SIN SUMAR MÉTRICAS)")
print("="*70)

event_count_dfs = []

for count_column in COLUMNS_TO_COUNT_ONLY:
    print(f"\n--- Contando por: {count_column} ---")
    
    if count_column not in df_all.columns:
        print(f"  ⚠️ Columna '{count_column}' no encontrada, saltando...")
        continue
    
    unique_values = df_all[count_column].apply(get_value_from_column).dropna().unique()
    unique_values = [v for v in unique_values if v]
    
    print(f"  Valores únicos encontrados: {len(unique_values)}")
    
    for value in unique_values:
        df_value = df_all[
            df_all[count_column].apply(get_value_from_column) == value
        ].copy()
        
        if len(df_value) == 0:
            continue
        
        # CONTAR eventos (no sumar métricas)
        value_counts = df_value.groupby("pid").size().reset_index(name="count")
        value_counts = value_counts.rename(columns={"pid": "player_id"})
        
        # Nombre de columna
        suffix = f"{count_column}_{str(value).lower().replace(' ', '_').replace('-', '_')}"
        col_name = f"n_events_{suffix}"
        
        value_counts = value_counts.rename(columns={"count": col_name})
        
        event_count_dfs.append(value_counts)
        
        discrimination_stats.append({
            "discrimination_type": "count_only",
            "column": count_column,
            "value": value,
            "suffix": col_name,
            "total_events": len(df_value),
            "unique_players": value_counts["player_id"].nunique()
        })
        
        print(f"  ✓ {value}: {len(df_value):,} eventos, {value_counts['player_id'].nunique()} jugadores")

print(f"\n✓ Total columnas de conteo generadas: {len(event_count_dfs)}")

# ============= 5B) DISCRIMINAR POR TERCIOS DE CANCHA =============
if ENABLE_THIRDS_ANALYSIS:
    print("\n" + "="*70)
    print("CREANDO MÉTRICAS DISCRIMINADAS POR TERCIOS DE CANCHA")
    print("="*70)
    
    # Verificar que tengamos coordenada X
    if "x" not in df_all.columns or df_all["x"].isna().all():
        print("⚠️ No hay coordenadas X disponibles. Saltando análisis por tercios.")
    else:
        pitch_length, pitch_width = infer_pitch_dimensions(df_all)
        print(f"Dimensiones inferidas: {pitch_length}m x {pitch_width}m")
        
        df_all["x_start"] = pd.to_numeric(df_all["x"], errors="coerce")
        df_all["third_start"] = df_all["x_start"].apply(lambda x: get_third_from_x(x, pitch_length))
        
        print(f"Eventos con tercio identificado: {df_all['third_start'].notna().sum():,}")
        
        metrics_for_thirds = [c for c in METRICS_FOR_THIRDS if c in num_cols]
        print(f"Métricas para análisis de tercios: {metrics_for_thirds}")
        
        for event_type in THIRDS_EVENTS:
            print(f"\n--- Procesando {event_type} por tercios ---")
            
            df_event = df_all[df_all["type"] == event_type].copy()
            
            if df_event.empty:
                print(f"  ⚠️ No hay eventos de tipo {event_type}")
                continue
            
            for third in ["defensive", "middle", "attacking"]:
                df_third = df_event[df_event["third_start"] == third].copy()
                
                if df_third.empty:
                    continue
                
                third_sums = df_third.groupby("pid")[metrics_for_thirds].sum(min_count=1).reset_index()
                third_sums = third_sums.rename(columns={"pid": "player_id"})
                
                event_name = event_type.lower().replace(" ", "_")
                suffix = f"third_{third}_{event_name}"
                
                rename_dict = {col: f"{col}_{suffix}" for col in metrics_for_thirds}
                third_sums = third_sums.rename(columns=rename_dict)
                
                all_discriminated_dfs[suffix] = third_sums
                
                discrimination_stats.append({
                    "discrimination_type": "third",
                    "column": "third",
                    "value": f"{third}_{event_name}",
                    "suffix": suffix,
                    "total_events": len(df_third),
                    "unique_players": third_sums["player_id"].nunique()
                })
                
                print(f"  ✓ {third}: {len(df_third):,} eventos, {third_sums['player_id'].nunique()} jugadores")
        
        # ============= 5B-SPECIAL) CROSS EN ÚLTIMO TERCIO (OPEN PLAY) =============
        if ENABLE_CROSS_ATTACKING:
            print("\n--- Procesando Pass Cross en último tercio (open play) ---")
            
            df_cross = df_all[df_all["type"] == "Pass"].copy()
            
            if "pass_cross" in df_cross.columns:
                df_cross["_pass_cross_bool"] = df_cross["pass_cross"].apply(
                    lambda x: str(x).lower() in ["true", "1", "1.0"] if pd.notna(x) else False
                )
                df_cross = df_cross[df_cross["_pass_cross_bool"] == True]
            
            df_cross = df_cross[df_cross["third_start"] == "attacking"]
            
            if "play_pattern" in df_cross.columns:
                df_cross["_play_pattern_clean"] = df_cross["play_pattern"].apply(get_value_from_column)
                df_cross = df_cross[df_cross["_play_pattern_clean"] == "Regular Play"]
            
            if not df_cross.empty:
                cross_sums = df_cross.groupby("pid")[metrics_for_thirds].sum(min_count=1).reset_index()
                cross_sums = cross_sums.rename(columns={"pid": "player_id"})
                
                suffix = "third_attacking_pass_cross_openplay"
                rename_dict = {col: f"{col}_{suffix}" for col in metrics_for_thirds}
                cross_sums = cross_sums.rename(columns=rename_dict)
                
                all_discriminated_dfs[suffix] = cross_sums
                
                discrimination_stats.append({
                    "discrimination_type": "third_special",
                    "column": "cross_attacking",
                    "value": "pass_cross_openplay",
                    "suffix": suffix,
                    "total_events": len(df_cross),
                    "unique_players": cross_sums["player_id"].nunique()
                })
                
                print(f"  ✓ Cross attacking (open play): {len(df_cross):,} eventos, {cross_sums['player_id'].nunique()} jugadores")
            else:
                print(f"  ⚠️ No se encontraron cross en último tercio (open play)")
        
        # ============= 5B-COUNT) CONTAR EVENTOS POR TERCIO =============
        print("\n" + "="*70)
        print("CONTANDO EVENTOS POR TERCIO")
        print("="*70)
        
        event_counts_by_third = []
        
        for event_type in THIRDS_EVENTS:
            df_event = df_all[df_all["type"] == event_type].copy()
            
            if df_event.empty:
                continue
            
            event_name = event_type.lower().replace(" ", "_")
            
            counts = df_event.groupby(["pid", "third_start"]).size().reset_index(name="count")
            
            for third in ["defensive", "middle", "attacking"]:
                counts_third = counts[counts["third_start"] == third].copy()
                
                if counts_third.empty:
                    continue
                
                col_name = f"n_events_third_{third}_{event_name}"
                counts_third = counts_third.rename(columns={"pid": "player_id", "count": col_name})
                counts_third = counts_third[["player_id", col_name]]
                
                event_counts_by_third.append(counts_third)
                
                total_count = counts_third[col_name].sum()
                print(f"  ✓ Contados {event_name} en {third}: {total_count:,} eventos totales")
        
        if ENABLE_CROSS_ATTACKING and 'df_cross' in locals() and not df_cross.empty:
            cross_counts = df_cross.groupby("pid").size().reset_index(name="n_events_third_attacking_pass_cross_openplay")
            cross_counts = cross_counts.rename(columns={"pid": "player_id"})
            event_counts_by_third.append(cross_counts)
            total_cross = cross_counts['n_events_third_attacking_pass_cross_openplay'].sum()
            print(f"  ✓ Contados cross attacking openplay: {total_cross:,} eventos totales")

# ============= 5C) SUMAR EVENTOS TOTALES =============
print("\n" + "="*70)
print("AGREGANDO EVENTOS TOTALES")
print("="*70)

player_sums = df_all.groupby("pid")[num_cols].sum(min_count=1).reset_index()
player_sums = player_sums.rename(columns={"pid": "player_id"})

player_sums = player_sums.merge(
    player_minutes_summary[[
        "player_id", "player_name", "primary_position", "all_positions", "teams",
        "total_minutes", "minutes_primary_position", "primary_position_share",
        "matches_played", "times_starter"
    ]], 
    on="player_id", 
    how="left"
)

# ============= 5C-TURNOVERS) AGREGAR MÉTRICAS DE TURNOVERS =============
if ENABLE_TURNOVER_ANALYSIS and 'turnover_counts' in locals():
    print("\n" + "="*70)
    print("INTEGRANDO MÉTRICAS DE TURNOVERS")
    print("="*70)
    
    player_sums = player_sums.merge(turnover_counts, on="player_id", how="left")
    
    # Llenar NaN con 0 para jugadores sin turnovers
    turnover_cols = [c for c in turnover_counts.columns if c != "player_id"]
    player_sums[turnover_cols] = player_sums[turnover_cols].fillna(0)
    
    print(f"✓ Métricas de turnovers integradas: {len(turnover_cols)} columnas")
    
    # Agregar turnovers a num_cols para que se calculen per90
    num_cols.extend(turnover_cols)

print(f"Jugadores en player_sums: {len(player_sums):,}")

# ============= 5D) MERGE DE TODAS LAS DISCRIMINACIONES =============
print("\n" + "="*70)
print("COMBINANDO TODAS LAS MÉTRICAS DISCRIMINADAS")
print("="*70)

discriminated_cols = []

# Merge discriminaciones con métricas (OBV, xG)
for suffix, disc_df in all_discriminated_dfs.items():
    player_sums = player_sums.merge(disc_df, on="player_id", how="left")
    new_cols = [c for c in disc_df.columns if c != "player_id"]
    discriminated_cols.extend(new_cols)

# Merge conteos simples de eventos
if event_count_dfs:
    print("\nIntegrando conteos de eventos (duel_outcome, interception_outcome)...")
    
    for counts_df in event_count_dfs:
        player_sums = player_sums.merge(counts_df, on="player_id", how="left")
        new_count_cols = [c for c in counts_df.columns if c != "player_id"]
        discriminated_cols.extend(new_count_cols)
    
    print(f"  ✓ Total conteos integrados: {len(event_count_dfs)}")

if ENABLE_THIRDS_ANALYSIS and 'event_counts_by_third' in locals() and event_counts_by_third:
    print("\nIntegrando contadores de eventos por tercio...")
    
    count_cols = []
    for counts_df in event_counts_by_third:
        player_sums = player_sums.merge(counts_df, on="player_id", how="left")
        new_count_cols = [c for c in counts_df.columns if c != "player_id"]
        count_cols.extend(new_count_cols)
    
    discriminated_cols.extend(count_cols)
    print(f"  ✓ Total contadores agregados: {len(count_cols)}")

print(f"\nTotal métricas discriminadas: {len(discriminated_cols)}")
print(f"  - Métricas de valores (sumas): {len([c for c in discriminated_cols if not c.startswith('n_events_')])}")
print(f"  - Contadores de eventos: {len([c for c in discriminated_cols if c.startswith('n_events_')])}")

player_sums[discriminated_cols] = player_sums[discriminated_cols].fillna(0)

# ============= 6) CALCULAR PORCENTAJES (DESDE TOTALES) =============
print("\n" + "="*70)
print("CALCULANDO MÉTRICAS DE PORCENTAJE")
print("="*70)

percentage_metrics = []

# Duel win percentage
if "duel_won" in player_sums.columns and "duel_lost" in player_sums.columns:
    player_sums["duel_win_pct"] = np.where(
        (player_sums["duel_won"] + player_sums["duel_lost"]) > 0,
        player_sums["duel_won"] / (player_sums["duel_won"] + player_sums["duel_lost"]),
        np.nan
    )
    percentage_metrics.append("duel_win_pct")
    print("✓ duel_win_pct")

# Tackle success percentage (usando conteos de duel_outcome)
print("\n--- Calculando tackle success % desde duel_outcome ---")

# Buscar columnas de conteo de duel_outcome
duel_won_cols = []
duel_lost_cols = []

for col in player_sums.columns:
    if "n_events_duel_outcome" in col:
        if "success_in_play" in col or "won" in col or "success_out" in col:
            duel_won_cols.append(col)
        elif "lost_in_play" in col or "lost_out" in col:
            duel_lost_cols.append(col)

if duel_won_cols and duel_lost_cols:
    print(f"  Columnas de duelos ganados: {duel_won_cols}")
    print(f"  Columnas de duelos perdidos: {duel_lost_cols}")
    
    # Sumar todos los duelos ganados
    player_sums["total_duels_won"] = sum(
        pd.to_numeric(player_sums[col], errors="coerce").fillna(0) 
        for col in duel_won_cols
    )
    
    # Sumar todos los duelos perdidos
    player_sums["total_duels_lost"] = sum(
        pd.to_numeric(player_sums[col], errors="coerce").fillna(0) 
        for col in duel_lost_cols
    )
    
    # Calcular porcentaje
    player_sums["tackle_success_pct"] = np.where(
        (player_sums["total_duels_won"] + player_sums["total_duels_lost"]) > 0,
        player_sums["total_duels_won"] / (player_sums["total_duels_won"] + player_sums["total_duels_lost"]),
        np.nan
    )
    
    percentage_metrics.append("tackle_success_pct")
    print("✓ tackle_success_pct calculado")
    
    # Estadísticas
    valid = player_sums["tackle_success_pct"].notna()
    if valid.any():
        avg_tackle = player_sums.loc[valid, "tackle_success_pct"].mean()
        print(f"  Promedio: {avg_tackle:.1%}")
else:
    print("⚠️  tackle_success_pct NO calculado (columnas n_events_duel_outcome no encontradas)")

# Ball recovery success percentage
if "ball_recovery_offensive" in player_sums.columns and "ball_recovery_recovery_failure" in player_sums.columns:
    player_sums["ball_recovery_success_pct"] = np.where(
        (player_sums["ball_recovery_offensive"] + player_sums["ball_recovery_recovery_failure"]) > 0,
        player_sums["ball_recovery_offensive"] / (player_sums["ball_recovery_offensive"] + player_sums["ball_recovery_recovery_failure"]),
        np.nan
    )
    percentage_metrics.append("ball_recovery_success_pct")
    print("✓ ball_recovery_success_pct")

# ============= PORCENTAJES DE DUELOS (desde outcome discriminado) =============
print("\n--- Calculando porcentajes de duelos desde outcomes ---")

# Duel success rate (desde n_events_duel_outcome_*)
duel_won_col = None
duel_lost_col = None

# Buscar columnas de conteo de duel_outcome
for col in player_sums.columns:
    if "n_events_duel_outcome" in col.lower():
        if "success_in_play" in col.lower() or "won" in col.lower():
            duel_won_col = col
        elif "lost_in_play" in col.lower() or "lost_out" in col.lower():
            if duel_lost_col is None:  # Solo tomar el primero
                duel_lost_col = col

if duel_won_col and duel_lost_col:
    # Convertir a numérico
    player_sums[duel_won_col] = pd.to_numeric(player_sums[duel_won_col], errors="coerce")
    player_sums[duel_lost_col] = pd.to_numeric(player_sums[duel_lost_col], errors="coerce")
    
    # Calcular porcentaje
    player_sums["duel_success_rate"] = np.where(
        (player_sums[duel_won_col] + player_sums[duel_lost_col]) > 0,
        player_sums[duel_won_col] / (player_sums[duel_won_col] + player_sums[duel_lost_col]),
        np.nan
    )
    percentage_metrics.append("duel_success_rate")
    print(f"✓ duel_success_rate (desde {duel_won_col} y {duel_lost_col})")
else:
    print(f"⚠️  No se encontraron columnas n_events_duel_outcome para calcular duel_success_rate")
    if duel_won_col:
        print(f"    Encontrado: {duel_won_col}")
    if duel_lost_col:
        print(f"    Encontrado: {duel_lost_col}")

# Aerial duel success rate (desde duel_type)
aerial_won_col = None
aerial_lost_col = None

for col in player_sums.columns:
    # Buscar en las métricas discriminadas de duel_type (estas SÍ tienen métricas sumadas)
    if "duel_type_aerial" in col.lower():
        # Estas columnas vienen de la discriminación con métricas, no de conteo
        # No las usamos aquí, buscamos en outcomes
        pass

# Buscar outcomes específicos de aerial
for col in player_sums.columns:
    if "n_events_duel_outcome" in col.lower() and "aerial" in col.lower():
        if "won" in col.lower() or "success" in col.lower():
            aerial_won_col = col
        elif "lost" in col.lower():
            aerial_lost_col = col

if aerial_won_col and aerial_lost_col:
    player_sums[aerial_won_col] = pd.to_numeric(player_sums[aerial_won_col], errors="coerce")
    player_sums[aerial_lost_col] = pd.to_numeric(player_sums[aerial_lost_col], errors="coerce")
    
    player_sums["aerial_duel_success_rate"] = np.where(
        (player_sums[aerial_won_col] + player_sums[aerial_lost_col]) > 0,
        player_sums[aerial_won_col] / (player_sums[aerial_won_col] + player_sums[aerial_lost_col]),
        np.nan
    )
    percentage_metrics.append("aerial_duel_success_rate")
    print(f"✓ aerial_duel_success_rate (desde {aerial_won_col} y {aerial_lost_col})")
else:
    print(f"⚠️  No se encontraron columnas aerial duel outcome para calcular aerial_duel_success_rate")

# ============= PORCENTAJES DE INTERCEPCIONES (desde outcome discriminado) =============
print("\n--- Calculando porcentajes de intercepciones desde outcomes ---")

interception_won_col = None
interception_lost_col = None

for col in player_sums.columns:
    if "n_events_interception_outcome" in col.lower():
        if "won" in col.lower() or "success" in col.lower():
            interception_won_col = col
        elif "lost_in_play" in col.lower():
            interception_lost_col = col
        elif "lost_out" in col.lower() and interception_lost_col is None:
            # Si no hay lost_in_play, usar lost_out como fallback
            interception_lost_col = col

if interception_won_col and interception_lost_col:
    player_sums[interception_won_col] = pd.to_numeric(player_sums[interception_won_col], errors="coerce")
    player_sums[interception_lost_col] = pd.to_numeric(player_sums[interception_lost_col], errors="coerce")
    
    player_sums["interception_success_rate"] = np.where(
        (player_sums[interception_won_col] + player_sums[interception_lost_col]) > 0,
        player_sums[interception_won_col] / (player_sums[interception_won_col] + player_sums[interception_lost_col]),
        np.nan
    )
    percentage_metrics.append("interception_success_rate")
    print(f"✓ interception_success_rate (desde {interception_won_col} y {interception_lost_col})")
else:
    print(f"⚠️  No se encontraron columnas n_events_interception_outcome para calcular interception_success_rate")
    if interception_won_col:
        print(f"    Encontrado: {interception_won_col}")
    if interception_lost_col:
        print(f"    Encontrado: {interception_lost_col}")

# ============= PORCENTAJE DE PASES COMPLETADOS (desde outcome discriminado) =============
print("\n--- Calculando porcentaje de pases completados desde outcomes ---")

pass_complete_col = None
pass_incomplete_col = None

# Buscar columnas de conteo de pass_outcome
for col in player_sums.columns:
    if "n_events_pass_outcome" in col.lower():
        # Pass completado (sin outcome = completado en StatsBomb)
        # O puede aparecer explícitamente
        if col.lower().endswith("_pass_outcome") or "complete" in col.lower():
            # Este es el caso donde pass_outcome está vacío (pase completado)
            # La columna se llamaría algo como "n_events_pass_outcome_"
            if col.count('_') == col.count('_'):  # Verificar si termina en outcome
                pass_complete_col = col
        # Pass incompleto
        if "incomplete" in col.lower() or "out" in col.lower() or "off_t" in col.lower():
            pass_incomplete_col = col

# En StatsBomb, si pass_outcome está vacío = pase completado
# Necesitamos contar pases totales vs pases con outcome (incompletos)

# Método alternativo: contar desde eventos de tipo Pass
print("  Método: Contando passes totales vs passes con outcome...")

# Contar todos los passes por jugador
passes_total_df = df_all[df_all["type"] == "Pass"].copy()
if not passes_total_df.empty:
    passes_total = passes_total_df.groupby("pid").size().reset_index(name="total_passes")
    passes_total = passes_total.rename(columns={"pid": "player_id"})
    
    # Merge con player_sums
    player_sums = player_sums.merge(passes_total, on="player_id", how="left")
    player_sums["total_passes"] = player_sums["total_passes"].fillna(0)
    
    # Contar pases incompletos (aquellos con pass_outcome no vacío)
    passes_incomplete_df = passes_total_df[passes_total_df["pass_outcome"].notna()].copy()
    
    if not passes_incomplete_df.empty:
        passes_incomplete = passes_incomplete_df.groupby("pid").size().reset_index(name="incomplete_passes")
        passes_incomplete = passes_incomplete.rename(columns={"pid": "player_id"})
        
        player_sums = player_sums.merge(passes_incomplete, on="player_id", how="left")
        player_sums["incomplete_passes"] = player_sums["incomplete_passes"].fillna(0)
    else:
        player_sums["incomplete_passes"] = 0
    
    # Calcular pases completados
    player_sums["complete_passes"] = player_sums["total_passes"] - player_sums["incomplete_passes"]
    
    # Calcular porcentaje
    player_sums["pass_completion_rate"] = np.where(
        player_sums["total_passes"] > 0,
        player_sums["complete_passes"] / player_sums["total_passes"],
        np.nan
    )
    
    percentage_metrics.append("pass_completion_rate")
    
    # Estadísticas
    valid = player_sums["pass_completion_rate"].notna()
    if valid.any():
        avg_completion = player_sums.loc[valid, "pass_completion_rate"].mean()
        players_with_passes = (player_sums["total_passes"] > 0).sum()
        total_passes_all = player_sums["total_passes"].sum()
        
        print(f"✓ pass_completion_rate calculado")
        print(f"  Total passes: {int(total_passes_all):,}")
        print(f"  Jugadores con passes: {players_with_passes:,}")
        print(f"  Promedio completion: {avg_completion:.1%}")
else:
    print("⚠️  No se encontraron eventos de tipo 'Pass'")
    player_sums["total_passes"] = 0
    player_sums["complete_passes"] = 0
    player_sums["incomplete_passes"] = 0
    player_sums["pass_completion_rate"] = np.nan

print(f"\nTotal métricas de porcentaje calculadas: {len(percentage_metrics)}")

# ============= 6B) CALCULAR TOTALES DE MÉTRICAS COMPUESTAS =============
print("\n" + "="*70)
print("CALCULANDO TOTALES DE MÉTRICAS COMPUESTAS")
print("="*70)

# Foul committed total (suma de sus variantes)
print("\n--- Calculando foul_committed total ---")
foul_cols = ["foul_committed_advantage", "foul_committed_penalty", "foul_committed_offensive"]
available_fouls = [c for c in foul_cols if c in player_sums.columns]

if available_fouls:
    print(f"  Columnas encontradas: {available_fouls}")
    player_sums["foul_committed"] = sum(
        pd.to_numeric(player_sums[col], errors="coerce").fillna(0) 
        for col in available_fouls
    )
    
    # Agregar a num_cols para que se normalice per90 automáticamente
    if "foul_committed" not in num_cols:
        num_cols.append("foul_committed")
    
    total = player_sums["foul_committed"].sum()
    players_with_fouls = (player_sums["foul_committed"] > 0).sum()
    print(f"✓ foul_committed calculado")
    print(f"  Total faltas: {int(total):,}")
    print(f"  Jugadores con faltas: {players_with_fouls:,}")
else:
    print(f"⚠️  No se encontraron columnas de foul_committed")
    player_sums["foul_committed"] = 0

# ============= 7) NORMALIZAR POR 90 MINUTOS =============
print("\n" + "="*70)
print("NORMALIZANDO POR 90 MINUTOS")
print("="*70)

for c in num_cols:
    player_sums[f"{c}_per90"] = np.where(
        player_sums["total_minutes"] > 0, 
        player_sums[c] / player_sums["total_minutes"] * 90.0, 
        np.nan
    )

for c in discriminated_cols:
    player_sums[f"{c}_per90"] = np.where(
        player_sums["total_minutes"] > 0, 
        player_sums[c] / player_sums["total_minutes"] * 90.0, 
        np.nan
    )

total_per90_cols = [f"{c}_per90" for c in num_cols]
discriminated_per90_cols = [f"{c}_per90" for c in discriminated_cols]

print(f"✓ Métricas totales per90: {len(total_per90_cols)}")
print(f"✓ Métricas discriminadas per90: {len(discriminated_per90_cols)}")

if ENABLE_TURNOVER_ANALYSIS:
    turnover_per90_cols = [c for c in total_per90_cols if 'turnover' in c]
    print(f"✓ Métricas de turnovers per90: {len(turnover_per90_cols)}")

# NOTA: La normalización per90 de shots/touches se hace después de calcularlos (Sección 10)
# No se hace aquí porque estas métricas aún no existen en player_sums

# ============= 8) CALCULAR MÉTRICAS COMPUESTAS (DESDE PER90) =============
print("\n" + "="*70)
print("CALCULANDO MÉTRICAS COMPUESTAS")
print("="*70)

composite_metrics = []

# Defensive actions lost per90 (usando conteos de duel_outcome + faltas)
print("\n--- Calculando defensive actions lost per90 ---")

defensive_components = []

# 1. Duelos perdidos (desde n_events_duel_outcome_lost)
duel_lost_per90_col = None
for col in player_sums.columns:
    if "n_events_duel_outcome_lost" in col and col.endswith("_per90"):
        duel_lost_per90_col = col
        break

if duel_lost_per90_col:
    defensive_components.append((duel_lost_per90_col, "duelos perdidos"))
    print(f"  ✓ Usando: {duel_lost_per90_col}")
else:
    # Si no existe per90, calcularlo desde el total
    lost_cols = [c for c in player_sums.columns if "n_events_duel_outcome_lost" in c and not c.endswith("_per90")]
    if lost_cols:
        # Sumar todos los duelos perdidos y normalizar
        player_sums["duels_lost_total"] = sum(
            pd.to_numeric(player_sums[col], errors="coerce").fillna(0) 
            for col in lost_cols
        )
        player_sums["duels_lost_per90"] = np.where(
            player_sums["total_minutes"] > 0,
            player_sums["duels_lost_total"] / player_sums["total_minutes"] * 90.0,
            np.nan
        )
        defensive_components.append(("duels_lost_per90", "duelos perdidos"))
        print(f"  ✓ Calculado: duels_lost_per90 (desde {len(lost_cols)} columnas)")

# 2. Faltas cometidas (calculado en sección 6B)
if "foul_committed_per90" in player_sums.columns:
    defensive_components.append(("foul_committed_per90", "faltas"))
    print(f"  ✓ Usando: foul_committed_per90")
else:
    print(f"  ⚠️  foul_committed_per90 no disponible")

# Calcular la suma
if defensive_components:
    player_sums["defensive_actions_lost_per90"] = sum(
        pd.to_numeric(player_sums[col], errors="coerce").fillna(0) 
        for col, _ in defensive_components
    )
    composite_metrics.append("defensive_actions_lost_per90")
    
    components_str = ", ".join([desc for _, desc in defensive_components])
    print(f"✓ defensive_actions_lost_per90 calculado ({components_str})")
    
    # Estadísticas
    avg = player_sums["defensive_actions_lost_per90"].mean()
    print(f"  Promedio: {avg:.2f} per90")
else:
    print(f"⚠️  defensive_actions_lost_per90 NO calculado (no se encontraron componentes)")

# Clearances total per90
clearance_cols = [
    "clearance_aerial_won_per90",
    "clearance_head_per90", 
    "clearance_left_foot_per90",
    "clearance_right_foot_per90",
    "clearance_other_per90"
]
available_clearances = [c for c in clearance_cols if c in player_sums.columns]
if available_clearances:
    player_sums["clearances_total_per90"] = sum(
        player_sums[c].fillna(0) for c in available_clearances
    )
    composite_metrics.append("clearances_total_per90")
    print(f"✓ clearances_total_per90 (from {len(available_clearances)} sources)")

# Blocks total per90
block_cols = ["block_deflection_per90", "block_save_block_per90"]
available_blocks = [c for c in block_cols if c in player_sums.columns]
if available_blocks:
    player_sums["blocks_total_per90"] = sum(
        player_sums[c].fillna(0) for c in available_blocks
    )
    composite_metrics.append("blocks_total_per90")
    print(f"✓ blocks_total_per90 (from {len(available_blocks)} sources)")

print(f"\nTotal métricas compuestas calculadas: {len(composite_metrics)}")

# ============= 9) CALCULAR XG POR SHOT =============
print("\n" + "="*70)
print("CALCULANDO XG POR SHOT")
print("="*70)

xg_per_shot_metrics = []

# Contar shots totales por jugador desde eventos
print("Contando shots por jugador...")
shots_df = df_all[df_all["type"] == "Shot"].copy()

if not shots_df.empty:
    shots_count = shots_df.groupby("pid").size().reset_index(name="total_shots")
    shots_count = shots_count.rename(columns={"pid": "player_id"})
    
    # Merge con player_sums
    player_sums = player_sums.merge(shots_count, on="player_id", how="left")
    player_sums["total_shots"] = player_sums["total_shots"].fillna(0)
    
    print(f"✓ Shots contados: {shots_count['total_shots'].sum():,.0f} total")
    print(f"✓ Jugadores con shots: {(player_sums['total_shots'] > 0).sum():,}")
    
    # Calcular xG por shot
    if "shot_statsbomb_xg" in player_sums.columns:
        player_sums["shot_statsbomb_xg"] = pd.to_numeric(player_sums["shot_statsbomb_xg"], errors="coerce")
        
        player_sums["xg_per_shot"] = np.where(
            player_sums["total_shots"] > 0,
            player_sums["shot_statsbomb_xg"] / player_sums["total_shots"],
            np.nan
        )
        
        xg_per_shot_metrics.append("xg_per_shot")
        
        # Estadísticas
        valid_xg = player_sums["xg_per_shot"].notna()
        avg_xg_per_shot = player_sums.loc[valid_xg, "xg_per_shot"].mean()
        
        print(f"✓ xg_per_shot calculado")
        print(f"  Promedio xG por shot: {avg_xg_per_shot:.3f}")
    else:
        print("⚠️  Columna 'shot_statsbomb_xg' no encontrada")
else:
    print("⚠️  No se encontraron eventos de tipo 'Shot'")
    player_sums["total_shots"] = 0
    player_sums["xg_per_shot"] = np.nan

print(f"\nTotal métricas xG por shot: {len(xg_per_shot_metrics)}")

# ============= 10) CALCULAR SHOT TOUCH % Y TOQUES EN ÁREA =============
print("\n" + "="*70)
print("CALCULANDO SHOT TOUCH % Y TOQUES EN ÁREA RIVAL")
print("="*70)

touch_metrics = []

# --- 1) CALCULAR TOQUES TOTALES ---
print("\n1. Contando toques totales por jugador...")

# En StatsBomb, casi todos los eventos son "toques" excepto algunos específicos
EXCLUDE_FROM_TOUCHES = {
    "Starting XI",
    "Half Start", 
    "Half End",
    "Substitution",
    "Player Off",
    "Player On",
    "Tactical Shift",
    "Injury Stoppage",
    "Bad Behaviour",
    "Referee Ball-Drop",
    "Shield",
    "Own Goal Against",
    "Own Goal For",
    "Error",
    "Offside",
}

# Eventos que SÍ cuentan como toques
df_touches = df_all[~df_all["type"].isin(EXCLUDE_FROM_TOUCHES)].copy()
touches_count = df_touches.groupby("pid").size().reset_index(name="total_touches")
touches_count = touches_count.rename(columns={"pid": "player_id"})

player_sums = player_sums.merge(touches_count, on="player_id", how="left")
player_sums["total_touches"] = player_sums["total_touches"].fillna(0)

print(f"✓ Toques contados: {touches_count['total_touches'].sum():,.0f} total")
print(f"✓ Jugadores con toques: {(player_sums['total_touches'] > 0).sum():,}")

# --- 2) CALCULAR SHOT TOUCH % ---
print("\n2. Calculando Shot Touch %...")

if "total_shots" in player_sums.columns and "total_touches" in player_sums.columns:
    player_sums["shot_touch_pct"] = np.where(
        player_sums["total_touches"] > 0,
        player_sums["total_shots"] / player_sums["total_touches"],
        np.nan
    )
    
    touch_metrics.append("shot_touch_pct")
    
    valid = player_sums["shot_touch_pct"].notna() & (player_sums["total_shots"] > 0)
    if valid.any():
        avg_shot_touch = player_sums.loc[valid, "shot_touch_pct"].mean()
        players_count = valid.sum()
        
        print(f"✓ shot_touch_pct calculado")
        print(f"  Promedio shot touch %: {avg_shot_touch:.2%}")
        print(f"  Jugadores con shots: {players_count:,}")
else:
    print("⚠️  No se pudo calcular shot_touch_pct (falta total_shots o total_touches)")

# --- 3) CALCULAR TOQUES EN ÁREA RIVAL ---
print("\n3. Contando toques en área rival...")

# Inferir dimensiones de la cancha
if "x" in df_all.columns and df_all["x"].notna().any():
    pitch_length, pitch_width = infer_pitch_dimensions(df_all)
    
    # Área rival en StatsBomb:
    # - X: desde (length - 18) hasta length (últimos 18 metros)
    # - Y: desde (width/2 - 22) hasta (width/2 + 22) (44 metros de ancho centrados)
    # Para cancha 120x80: x >= 102, 18 <= y <= 62
    
    penalty_box_x_min = pitch_length - 18.0
    penalty_box_y_min = (pitch_width / 2.0) - 22.0  # 40 - 22 = 18
    penalty_box_y_max = (pitch_width / 2.0) + 22.0  # 40 + 22 = 62
    
    print(f"  Dimensiones cancha: {pitch_length}m x {pitch_width}m")
    print(f"  Área rival: X >= {penalty_box_x_min:.1f}, {penalty_box_y_min:.1f} <= Y <= {penalty_box_y_max:.1f}")
    
    # Filtrar eventos dentro del área rival
    df_in_box = df_touches[
        (df_touches["x"] >= penalty_box_x_min) &
        (df_touches["y"] >= penalty_box_y_min) &
        (df_touches["y"] <= penalty_box_y_max)
    ].copy()
    
    if not df_in_box.empty:
        touches_in_box = df_in_box.groupby("pid").size().reset_index(name="touches_in_opp_box")
        touches_in_box = touches_in_box.rename(columns={"pid": "player_id"})
        
        player_sums = player_sums.merge(touches_in_box, on="player_id", how="left")
        player_sums["touches_in_opp_box"] = player_sums["touches_in_opp_box"].fillna(0)
        
        touch_metrics.append("touches_in_opp_box")
        
        # Calcular también el porcentaje de toques en área
        player_sums["touches_in_opp_box_pct"] = np.where(
            player_sums["total_touches"] > 0,
            player_sums["touches_in_opp_box"] / player_sums["total_touches"],
            np.nan
        )
        
        touch_metrics.append("touches_in_opp_box_pct")
        
        # Estadísticas
        total_in_box = touches_in_box["touches_in_opp_box"].sum()
        players_in_box = (player_sums["touches_in_opp_box"] > 0).sum()
        
        print(f"✓ touches_in_opp_box contados: {int(total_in_box):,} toques")
        print(f"✓ Jugadores con toques en área: {players_in_box:,}")
        
        valid_pct = player_sums["touches_in_opp_box_pct"].notna() & (player_sums["touches_in_opp_box"] > 0)
        if valid_pct.any():
            avg_pct = player_sums.loc[valid_pct, "touches_in_opp_box_pct"].mean()
            print(f"✓ Promedio % toques en área: {avg_pct:.2%}")
    else:
        print("⚠️  No se encontraron toques en área rival")
        player_sums["touches_in_opp_box"] = 0
        player_sums["touches_in_opp_box_pct"] = np.nan
else:
    print("⚠️  No hay coordenadas X/Y disponibles para calcular toques en área")
    player_sums["touches_in_opp_box"] = 0
    player_sums["touches_in_opp_box_pct"] = np.nan

print(f"\n✓ Total métricas de toques calculadas: {len(touch_metrics)}")

# Normalizar métricas de shots y toques per90 (AHORA QUE YA EXISTEN)
print("\n--- Normalizando métricas de shots y toques per90 ---")

shots_touches_per90 = []

for metric in ["total_shots", "total_touches", "touches_in_opp_box", "complete_passes"]:
    if metric in player_sums.columns:
        player_sums[metric] = pd.to_numeric(player_sums[metric], errors="coerce")
        player_sums[f"{metric}_per90"] = np.where(
            player_sums["total_minutes"] > 0,
            player_sums[metric] / player_sums["total_minutes"] * 90.0,
            np.nan
        )
        shots_touches_per90.append(f"{metric}_per90")
        print(f"✓ {metric}_per90")
    else:
        print(f"⚠️  {metric} no encontrado en player_sums")

print(f"✓ Métricas de shots/toques per90: {len(shots_touches_per90)}")

# ============= 11) PREPARAR DATAFRAME FINAL =============
base_cols = [
    "player_id", "player_name", "primary_position", "all_positions", "teams",
    "total_minutes", "minutes_primary_position", "primary_position_share",
    "matches_played", "times_starter"
]

final_cols = (base_cols + 
              num_cols + 
              percentage_metrics +
              discriminated_cols + 
              total_per90_cols + 
              discriminated_per90_cols +
              composite_metrics +
              xg_per_shot_metrics +
              touch_metrics +
              shots_touches_per90)

# Validar qué columnas existen
print("\n--- Validando columnas para export ---")
print(f"Total columnas en final_cols: {len(final_cols)}")

# Verificar columnas clave
key_cols_to_check = [
    "touches_in_opp_box_per90",
    "total_shots_per90", 
    "total_touches_per90",
    "complete_passes_per90",
    "tackle_success_pct",
    "defensive_actions_lost_per90"
]

print("\nColumnas clave:")
for col in key_cols_to_check:
    in_final_cols = col in final_cols
    in_player_sums = col in player_sums.columns
    print(f"  {col}:")
    print(f"    En final_cols: {in_final_cols}")
    print(f"    En player_sums: {in_player_sums}")
    if in_final_cols and not in_player_sums:
        print(f"    ⚠️  PROBLEMA: En final_cols pero NO en player_sums")

# Filtrar solo columnas que existen en player_sums
missing_cols = [c for c in final_cols if c not in player_sums.columns]
if missing_cols:
    print(f"\n⚠️  Columnas en final_cols que NO existen en player_sums: {len(missing_cols)}")
    print(f"Primeras 10: {missing_cols[:10]}")
    final_cols = [c for c in final_cols if c in player_sums.columns]
    print(f"✓ Filtradas. Columnas finales: {len(final_cols)}")

final_df = player_sums[final_cols].copy()
final_df = final_df.sort_values("total_minutes", ascending=False)

# ============= 11B) MÉTRICAS DE CARRILES OBV =============
if OBV_LANES_AVAILABLE:
    print("\n" + "="*70)
    print("CALCULANDO MÉTRICAS DE CARRILES Y ZONAS OBV")
    print("="*70)
    
    try:
        # Verificar que df_all existe y tiene las columnas necesarias
        if 'df_all' not in locals() and 'df_all' not in globals():
            print("  ⚠️  df_all no existe. Usando df en su lugar.")
            df_for_lanes = df
        else:
            df_for_lanes = df_all
        
        # Verificar columnas necesarias
        required_cols = ['pid', 'type', 'y', 'pass_end_location', 'obv_total_net']
        missing = [c for c in required_cols if c not in df_for_lanes.columns]
        
        if missing:
            print(f"  ⚠️  Columnas faltantes en df_for_lanes: {missing}")
            if 'pid' in missing:
                print(f"  ⚠️  Creando columna 'pid' en df_for_lanes...")
                df_for_lanes = df_for_lanes.copy()
                df_for_lanes["pid"] = get_player_id_series(df_for_lanes).astype("Int64")
        
        print(f"  DataFrame para carriles: {len(df_for_lanes):,} eventos")
        print(f"  Columnas disponibles: pid={('pid' in df_for_lanes.columns)}, type={('type' in df_for_lanes.columns)}")
        
        # Obtener ancho de cancha (ya inferido)
        _, pitch_width = infer_pitch_dimensions(df)
        
        print(f"  Columnas en final_df ANTES del merge: {len(final_df.columns)}")
        print(f"  Filas en final_df: {len(final_df)}")
        
        # Calcular métricas de carriles para TODOS los jugadores
        lanes_metrics = calculate_lane_bias_metrics(
            df=df_for_lanes,                                # df_all o df con 'pid'
            player_minutes_summary=player_minutes_summary,  # Ya calculado
            pitch_width=pitch_width,                        # Ya inferido
            extract_x_y_func=extract_x_y_from_location,    # Función helper
            min_passes=50,                                  # Mínimo de pases para calcular bias
        )
        
        if not lanes_metrics.empty:
            print(f"  Filas en lanes_metrics: {len(lanes_metrics)}")
            print(f"  Columnas en lanes_metrics: {list(lanes_metrics.columns)}")
            
            # Verificar si hay player_id en común
            common_ids = set(final_df["player_id"]) & set(lanes_metrics["player_id"])
            print(f"  Player IDs en común: {len(common_ids)}")
            
            # Merge con final_df
            final_df = final_df.merge(
                lanes_metrics,
                on="player_id",
                how="left"
            )
            
            print(f"  Columnas en final_df DESPUÉS del merge: {len(final_df.columns)}")
            print(f"Métricas de carriles agregadas a final_df")
            print(f"  Jugadores con lane_bias_index: {lanes_metrics['lane_bias_index'].notna().sum():,}")
            print(f"  Columnas agregadas: {list(lanes_metrics.columns[1:])}")
            
            # Verificar que las columnas están en final_df
            for col in lanes_metrics.columns[1:]:
                if col in final_df.columns:
                    non_null = final_df[col].notna().sum()
                    print(f"    - {col}: {non_null} valores no nulos")
                else:
                    print(f"    ⚠️  {col}: NO ESTÁ en final_df")
        else:
            print(f"\n⚠️  No se generaron métricas de carriles (lanes_metrics está vacío)")
    
    except Exception as e:
        print(f"\n❌ Error calculando métricas de carriles: {e}")
        import traceback
        traceback.print_exc()
else:
    print("\n⚠️  Módulo obv_lanes no disponible. Métricas de carriles omitidas.")

if CB_ZONE_AVAILABLE:
    try:
        zone_metrics = calculate_cb_zone_metrics(
            df=df_all,
            player_minutes_summary=player_minutes_summary,
            pitch_length=120.0,
            pitch_width=80.0,
        )
        if not zone_metrics.empty:
            final_df = final_df.merge(zone_metrics, on="player_id", how="left")
            print(f"Métricas de zona agregadas: {len(zone_metrics):,} jugadores")
    except Exception as e:
        print(f"❌ Error en cb_zone_builder: {e}")

# ============= 11C) MÉTRICAS DE ARQUEROS =============
if GK_METRICS_AVAILABLE:
    print("\n" + "="*70)
    print("CALCULANDO MÉTRICAS DE ARQUEROS")
    print("="*70)

    try:
        # Verificar que df_all existe
        if 'df_all' not in dir():
            df_for_gk = df
        else:
            df_for_gk = df_all

        print(f"  DataFrame para arqueros: {len(df_for_gk):,} eventos")

        # Calcular métricas de arqueros
        gk_metrics = calculate_gk_metrics(
            df=df_for_gk,
            player_minutes_summary=player_minutes_summary,
        )

        if not gk_metrics.empty:
            # Verificar player_ids en común
            common_ids = set(final_df["player_id"]) & set(gk_metrics["player_id"])
            print(f"  Player IDs en común con final_df: {len(common_ids)}")

            # Merge con final_df
            final_df = final_df.merge(
                gk_metrics,
                on="player_id",
                how="left"
            )

            print(f"  Columnas en final_df después del merge: {len(final_df.columns)}")
            print(f"Métricas de arqueros agregadas a final_df")

            # Resumen de valores no nulos por métrica
            gk_cols = [c for c in gk_metrics.columns if c != "player_id"]
            for col in gk_cols:
                if col in final_df.columns:
                    non_null = final_df[col].notna().sum()
                    print(f"    - {col}: {non_null} valores no nulos")
        else:
            print("\n⚠️  No se generaron métricas de arqueros (DataFrame vacío)")

    except Exception as e:
        print(f"\n❌ Error calculando métricas de arqueros: {e}")
        import traceback
        traceback.print_exc()
else:
    print("\n⚠️  Módulo goalkeeper_metrics no disponible. Métricas de arqueros omitidas.")

# ============= 12) EXPORTS CON TEMPORADA EN NOMBRE =============
print("\n" + "="*70)
print("EXPORTANDO RESULTADOS")
print("="*70)

# ============= ÚNICO EXPORT CSV: all_players_complete =============
output_complete = OUTPUTS_DIR / f"all_players_complete_{season}.csv"
final_df.to_csv(output_complete, index=False, encoding="utf-8-sig")
print(f"\n✓ CSV EXPORTADO: {output_complete}")

# ============= EXPORT: player_minutes_by_match =============
output_minutes = OUTPUTS_DIR / f"player_minutes_by_match_{season}.csv"
minutes_by_match_export = minutes_df.merge(
    player_minutes_summary[["player_id", "player_name", "primary_position"]],
    on="player_id", how="left"
)
minutes_by_match_export.to_csv(output_minutes, index=False)
print(f"✓ CSV EXPORTADO: {output_minutes} ({len(minutes_by_match_export):,} registros)")

# ============= RESTO DE OUTPUTS: SOLO EN LOG (NO CSV) =============
print("\n" + "="*70)
print("OUTPUTS GENERADOS (SOLO EN LOG - NO SE EXPORTAN COMO CSV)")
print("="*70)

# 2. Solo per90 (totales + discriminadas) - IMPRESO EN LOG
print("\n--- ALL PLAYERS PER90 ALL ---")
per90_all_cols = base_cols + total_per90_cols + discriminated_per90_cols
per90_all_df = final_df[per90_all_cols].copy()
print(f"Columnas: {len(per90_all_df.columns)}")
print(f"Jugadores: {len(per90_all_df)}")
print("\nPrimeras 3 columnas y 3 filas:")
print(per90_all_df.iloc[:3, :3].to_string())

# 3. Solo discriminadas per90 - IMPRESO EN LOG
print("\n--- ALL PLAYERS PER90 DISCRIMINATED ---")
disc_per90_cols = base_cols + discriminated_per90_cols
disc_per90_df = final_df[disc_per90_cols].copy()
print(f"Columnas: {len(disc_per90_df.columns)}")
print(f"Jugadores: {len(disc_per90_df)}")

# 4. Estadísticas de discriminaciones - IMPRESO EN LOG
print("\n--- DISCRIMINATION STATISTICS ---")
disc_stats_df = pd.DataFrame(discrimination_stats).sort_values("total_events", ascending=False)
print(f"Total discriminaciones: {len(disc_stats_df)}")
print("\nTop 10 discriminaciones por volumen:")
print(disc_stats_df.head(10).to_string(index=False))

# 5. Solo tercios per90 - IMPRESO EN LOG
if ENABLE_THIRDS_ANALYSIS:
    print("\n--- ALL PLAYERS THIRDS PER90 ---")
    thirds_cols = [c for c in discriminated_per90_cols if "third_" in c]
    if thirds_cols:
        thirds_df = final_df[base_cols + thirds_cols].copy()
        print(f"Columnas: {len(thirds_df.columns)}")
        print(f"Jugadores: {len(thirds_df)}")

# 6. Minutos por partido CON POSICIONES - IMPRESO EN LOG
print("\n--- PLAYER MINUTES BY MATCH ---")
minutes_with_pos = minutes_df.merge(
    player_minutes_summary[["player_id", "player_name", "primary_position"]], 
    on="player_id", how="left"
)
print(f"Total registros: {len(minutes_with_pos)}")
print("\nPrimeras 5 filas:")
print(minutes_with_pos.head(5).to_string(index=False))

# 7. Minutos por jugador y posición - IMPRESO EN LOG
print("\n--- PLAYER MINUTES BY POSITION ---")
pos_minutes_detail = position_minutes.merge(
    player_minutes_summary[["player_id", "player_name", "total_minutes"]], 
    on="player_id", how="left"
).sort_values(["player_id", "minutes_in_position"], ascending=[True, False])
print(f"Total registros: {len(pos_minutes_detail)}")
print("\nPrimeras 5 filas:")
print(pos_minutes_detail.head(5).to_string(index=False))

# 8. Detalle de turnovers - IMPRESO EN LOG
if ENABLE_TURNOVER_ANALYSIS and 'turnovers_df' in locals() and not turnovers_df.empty:
    print("\n--- TURNOVERS DETAIL ---")
    print(f"Total eventos de turnover: {len(turnovers_df)}")
    print("\nPrimeras 5 filas:")
    print(turnovers_df.head(5).to_string(index=False))
    
    # 9. Solo métricas de turnovers per90 - IMPRESO EN LOG
    if 'turnover_per90_cols' in locals() and turnover_per90_cols:
        print("\n--- ALL PLAYERS TURNOVERS PER90 ---")
        turnovers_per90_df = final_df[base_cols + turnover_per90_cols].copy()
        print(f"Columnas: {len(turnovers_per90_df.columns)}")
        print(f"Jugadores: {len(turnovers_per90_df)}")

# 10. Log del análisis
print(f"\n✓ Log completo del análisis guardado en: {log_filename}")

# ============= 13) RESUMEN FINAL =============
print("\n" + "="*70)
print(f"RESUMEN FINAL - TEMPORADA {season}")
print("="*70)
print(f"Jugadores procesados: {len(final_df):,}")
print(f"Partidos únicos: {df['match_id'].nunique():,}")
print(f"Total eventos: {len(df):,}")
print(f"Métricas base: {len(num_cols)}")
print(f"Columnas discriminadas: {len(COLUMNS_TO_DISCRIMINATE)}")
if ENABLE_THIRDS_ANALYSIS:
    print(f"Eventos analizados por tercios: {len(THIRDS_EVENTS)}")
if ENABLE_TURNOVER_ANALYSIS and 'turnovers_df' in locals():
    print(f"Turnovers detectados: {len(turnovers_df):,}")
print(f"Métricas discriminadas generadas: {len(discriminated_cols)}")
print(f"Total columnas finales: {len(final_df.columns):,}")

print(f"\n📊 Análisis de posiciones:")
print(f"  Jugadores con posición principal: {final_df['primary_position'].notna().sum()}")
print(f"  Promedio de especialización (% en posición principal): {final_df['primary_position_share'].mean():.1%}")

if percentage_metrics:
    print(f"\n📈 Métricas de porcentaje calculadas: {len(percentage_metrics)}")
    print(f"\n  Porcentajes básicos:")
    for metric in ["duel_win_pct", "tackle_success_pct", "ball_recovery_success_pct"]:
        if metric in percentage_metrics and metric in final_df.columns:
            avg = final_df[metric].mean()
            if not np.isnan(avg):
                print(f"    - {metric}: promedio = {avg:.1%}")
    
    print(f"\n  Porcentajes de duelos (desde outcomes):")
    for metric in ["duel_success_rate", "aerial_duel_success_rate"]:
        if metric in percentage_metrics and metric in final_df.columns:
            avg = final_df[metric].mean()
            count = final_df[metric].notna().sum()
            if not np.isnan(avg):
                print(f"    - {metric}: promedio = {avg:.1%} ({count} jugadores)")
    
    print(f"\n  Porcentajes de intercepciones (desde outcomes):")
    for metric in ["interception_success_rate"]:
        if metric in percentage_metrics and metric in final_df.columns:
            avg = final_df[metric].mean()
            count = final_df[metric].notna().sum()
            if not np.isnan(avg):
                print(f"    - {metric}: promedio = {avg:.1%} ({count} jugadores)")
    
    print(f"\n  Porcentaje de pases completados:")
    if "pass_completion_rate" in percentage_metrics and "pass_completion_rate" in final_df.columns:
        valid = final_df["pass_completion_rate"].notna()
        if valid.any():
            avg = final_df.loc[valid, "pass_completion_rate"].mean()
            count = valid.sum()
            print(f"    - pass_completion_rate: promedio = {avg:.1%} ({count} jugadores)")
            
            if "total_passes" in final_df.columns:
                total = final_df["total_passes"].sum()
                print(f"    - total_passes: {int(total):,}")
            if "complete_passes" in final_df.columns:
                complete = final_df["complete_passes"].sum()
                print(f"    - complete_passes: {int(complete):,}")
            if "incomplete_passes" in final_df.columns:
                incomplete = final_df["incomplete_passes"].sum()
                print(f"    - incomplete_passes: {int(incomplete):,}")

if composite_metrics:
    print(f"\n🔢 Métricas compuestas calculadas: {len(composite_metrics)}")
    for metric in composite_metrics:
        if metric in final_df.columns:
            print(f"  - {metric}")

if xg_per_shot_metrics:
    print(f"\n⚽ Métricas de xG por shot:")
    if "xg_per_shot" in final_df.columns:
        valid = final_df["xg_per_shot"].notna()
        if valid.any():
            avg = final_df.loc[valid, "xg_per_shot"].mean()
            median = final_df.loc[valid, "xg_per_shot"].median()
            players = valid.sum()
            print(f"  - xg_per_shot:")
            print(f"    Promedio: {avg:.3f}")
            print(f"    Mediana: {median:.3f}")
            print(f"    Jugadores: {players:,}")
    
    if "total_shots" in final_df.columns:
        total = final_df["total_shots"].sum()
        players_with_shots = (final_df["total_shots"] > 0).sum()
        print(f"  - total_shots:")
        print(f"    Total: {int(total):,}")
        print(f"    Jugadores con shots: {players_with_shots:,}")

if touch_metrics:
    print(f"\n🤾 Métricas de toques:")
    
    if "total_touches" in final_df.columns:
        total_touches = final_df["total_touches"].sum()
        players_touches = (final_df["total_touches"] > 0).sum()
        print(f"  - total_touches:")
        print(f"    Total: {int(total_touches):,}")
        print(f"    Jugadores: {players_touches:,}")
        
        if "total_touches_per90" in final_df.columns:
            avg_per90 = final_df["total_touches_per90"].mean()
            print(f"    Promedio per90: {avg_per90:.1f}")
    
    if "shot_touch_pct" in final_df.columns:
        valid = final_df["shot_touch_pct"].notna() & (final_df["shot_touch_pct"] > 0)
        if valid.any():
            avg = final_df.loc[valid, "shot_touch_pct"].mean()
            median = final_df.loc[valid, "shot_touch_pct"].median()
            count = valid.sum()
            print(f"  - shot_touch_pct:")
            print(f"    Promedio: {avg:.2%}")
            print(f"    Mediana: {median:.2%}")
            print(f"    Jugadores: {count:,}")
    
    if "total_shots" in final_df.columns and "total_shots_per90" in final_df.columns:
        total_shots = final_df["total_shots"].sum()
        players_shots = (final_df["total_shots"] > 0).sum()
        avg_shots_per90 = final_df.loc[final_df["total_shots"] > 0, "total_shots_per90"].mean()
        print(f"  - total_shots:")
        print(f"    Total: {int(total_shots):,}")
        print(f"    Jugadores: {players_shots:,}")
        print(f"    Promedio per90: {avg_shots_per90:.2f}")
    
    if "touches_in_opp_box" in final_df.columns:
        total_box = final_df["touches_in_opp_box"].sum()
        players_box = (final_df["touches_in_opp_box"] > 0).sum()
        print(f"  - touches_in_opp_box:")
        print(f"    Total: {int(total_box):,}")
        print(f"    Jugadores: {players_box:,}")
        
        if "touches_in_opp_box_per90" in final_df.columns:
            avg_box_per90 = final_df.loc[final_df["touches_in_opp_box"] > 0, "touches_in_opp_box_per90"].mean()
            print(f"    Promedio per90: {avg_box_per90:.2f}")
        
        if "touches_in_opp_box_pct" in final_df.columns:
            valid_pct = final_df["touches_in_opp_box_pct"].notna() & (final_df["touches_in_opp_box"] > 0)
            if valid_pct.any():
                avg_pct = final_df.loc[valid_pct, "touches_in_opp_box_pct"].mean()
                print(f"  - touches_in_opp_box_pct (promedio): {avg_pct:.2%}")

print(f"\nTop 10 posiciones principales más comunes:")
if final_df["primary_position"].notna().any():
    print(final_df["primary_position"].value_counts().head(10).to_string())

print(f"\nTop 20 discriminaciones por eventos:")
print(disc_stats_df.head(20)[["discrimination_type", "column", "value", "total_events", "unique_players"]].to_string(index=False))

if ENABLE_THIRDS_ANALYSIS:
    thirds_stats = disc_stats_df[disc_stats_df["discrimination_type"].isin(["third", "third_special"])]
    if not thirds_stats.empty:
        print(f"\nEstadísticas de tercios:")
        print(thirds_stats[["value", "total_events", "unique_players"]].to_string(index=False))

if ENABLE_TURNOVER_ANALYSIS and 'turnovers_df' in locals():
    print(f"\n🔄 Estadísticas de turnovers:")
    print(f"  Total turnovers: {len(turnovers_df):,}")
    print(f"  Jugadores con turnovers: {turnovers_df['player_id'].nunique():,}")
    print(f"  Promedio por jugador: {len(turnovers_df) / turnovers_df['player_id'].nunique():.1f}")
    
    if 'turnover_how' in turnovers_df.columns:
        print(f"\n  Top 5 tipos de turnovers:")
        for idx, (how, count) in enumerate(turnovers_df['turnover_how'].value_counts().head(5).items(), 1):
            print(f"    {idx}. {how}: {count:,} ({count/len(turnovers_df)*100:.1f}%)")

if into_final_third_metrics:
    print(f"\n⚽ Métricas 'into final third':")
    for metric in into_final_third_metrics:
        if metric in final_df.columns:
            total = final_df[metric].sum()
            players = (final_df[metric] > 0).sum()
            avg_per90 = final_df[f"{metric}_per90"].mean()
            print(f"  {metric}:")
            print(f"    Total eventos: {int(total):,}")
            print(f"    Jugadores: {players:,}")
            print(f"    Promedio per90: {avg_per90:.2f}")

print("="*70)
print(f"\nANÁLISIS COMPLETADO EXITOSAMENTE")
print(f"Fin: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print(f"Log guardado en: {log_filename}")
print("="*70)

# Cerrar el logger
sys.stdout.close()
sys.stdout = sys.stdout.terminal  # Restaurar stdout original