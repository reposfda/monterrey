# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import json, ast
import sys
from datetime import datetime

# Importar funciones de turnover
from turnover_scoring import compute_player_turnovers

# ============= CONFIG =============
PATH = "data/events_2024_2025.csv"
season = PATH.split('_', 1)[1].replace('.csv', '')

ASSUME_END_CAP = 120
OUTPUT_DIR = "./outputs"

# ============= LOGGING SETUP =============
class Logger:
    """Clase para escribir simultáneamente en consola y archivo"""
    def __init__(self, filename):
        self.terminal = sys.stdout
        self.log = open(filename, 'w', encoding='utf-8')
        
    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
        
    def flush(self):
        self.terminal.flush()
        self.log.flush()
        
    def close(self):
        self.log.close()

# Crear directorio de outputs si no existe
import os
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Configurar logging
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
log_filename = os.path.join(OUTPUT_DIR, f"analysis_log_{season}_{timestamp}.txt")
sys.stdout = Logger(log_filename)

print("="*70)
print(f"ANÁLISIS DE EVENTOS - TEMPORADA {season}")
print(f"Inicio: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print("="*70)

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

# ============= CONFIG TERCIOS DE CANCHA =============
ENABLE_THIRDS_ANALYSIS = True

THIRDS_EVENTS = [
    "Carry",
    "Pass",
    "Ball Recovery",
    "Duel",
    "Interception",
    "Pressure",
    "Dispossessed"
]

ENABLE_CROSS_ATTACKING = True

# ============= CONFIG TURNOVERS =============
ENABLE_TURNOVER_ANALYSIS = True
TURNOVER_OPEN_PLAY_ONLY = True
TURNOVER_EXCLUDE_RESTARTS = True

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

for suffix, disc_df in all_discriminated_dfs.items():
    player_sums = player_sums.merge(disc_df, on="player_id", how="left")
    new_cols = [c for c in disc_df.columns if c != "player_id"]
    discriminated_cols.extend(new_cols)

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

# Tackle success percentage
if "duel_tackle" in player_sums.columns and "duel_tackle_lost" in player_sums.columns:
    player_sums["tackle_success_pct"] = np.where(
        (player_sums["duel_tackle"] + player_sums["duel_tackle_lost"]) > 0,
        player_sums["duel_tackle"] / (player_sums["duel_tackle"] + player_sums["duel_tackle_lost"]),
        np.nan
    )
    percentage_metrics.append("tackle_success_pct")
    print("✓ tackle_success_pct")

# Ball recovery success percentage
if "ball_recovery_offensive" in player_sums.columns and "ball_recovery_recovery_failure" in player_sums.columns:
    player_sums["ball_recovery_success_pct"] = np.where(
        (player_sums["ball_recovery_offensive"] + player_sums["ball_recovery_recovery_failure"]) > 0,
        player_sums["ball_recovery_offensive"] / (player_sums["ball_recovery_offensive"] + player_sums["ball_recovery_recovery_failure"]),
        np.nan
    )
    percentage_metrics.append("ball_recovery_success_pct")
    print("✓ ball_recovery_success_pct")

print(f"\nTotal métricas de porcentaje calculadas: {len(percentage_metrics)}")

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

# ============= 8) CALCULAR MÉTRICAS COMPUESTAS (DESDE PER90) =============
print("\n" + "="*70)
print("CALCULANDO MÉTRICAS COMPUESTAS")
print("="*70)

composite_metrics = []

# Defensive actions lost per90
required_defensive = ["dribbled_past_per90", "duel_tackle_lost_per90", "foul_committed_per90"]
if all(c in player_sums.columns for c in required_defensive):
    player_sums["defensive_actions_lost_per90"] = (
        player_sums["dribbled_past_per90"].fillna(0)
        + player_sums["duel_tackle_lost_per90"].fillna(0)
        + player_sums["foul_committed_per90"].fillna(0)
    )
    composite_metrics.append("defensive_actions_lost_per90")
    print("✓ defensive_actions_lost_per90")

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

# ============= 9) PREPARAR DATAFRAME FINAL =============
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
              composite_metrics)

final_df = player_sums[final_cols].copy()
final_df = final_df.sort_values("total_minutes", ascending=False)

# ============= 10) EXPORTS CON TEMPORADA EN NOMBRE =============
print("\n" + "="*70)
print("EXPORTANDO RESULTADOS")
print("="*70)

# 1. Completo
output_complete = os.path.join(OUTPUT_DIR, f"all_players_complete_{season}.csv")
final_df.to_csv(output_complete, index=False)
print(f"✓ Completo: {output_complete}")

# 2. Solo per90 (totales + discriminadas)
output_per90_all = os.path.join(OUTPUT_DIR, f"all_players_per90_all_{season}.csv")
per90_all_cols = base_cols + total_per90_cols + discriminated_per90_cols
final_df[per90_all_cols].to_csv(output_per90_all, index=False)
print(f"✓ Per90 (todas): {output_per90_all}")

# 3. Solo discriminadas per90
output_per90_disc = os.path.join(OUTPUT_DIR, f"all_players_per90_discriminated_{season}.csv")
disc_per90_cols = base_cols + discriminated_per90_cols
final_df[disc_per90_cols].to_csv(output_per90_disc, index=False)
print(f"✓ Per90 (solo discriminadas): {output_per90_disc}")

# 4. Estadísticas de discriminaciones
disc_stats_df = pd.DataFrame(discrimination_stats).sort_values("total_events", ascending=False)
output_stats = os.path.join(OUTPUT_DIR, f"discrimination_statistics_{season}.csv")
disc_stats_df.to_csv(output_stats, index=False)
print(f"✓ Estadísticas: {output_stats}")

# 5. Solo tercios per90
if ENABLE_THIRDS_ANALYSIS:
    thirds_cols = [c for c in discriminated_per90_cols if "third_" in c]
    if thirds_cols:
        output_thirds = os.path.join(OUTPUT_DIR, f"all_players_thirds_per90_{season}.csv")
        final_df[base_cols + thirds_cols].to_csv(output_thirds, index=False)
        print(f"✓ Solo tercios per90: {output_thirds}")

# 6. Minutos por partido CON POSICIONES
output_minutes = os.path.join(OUTPUT_DIR, f"player_minutes_by_match_{season}.csv")
minutes_df.merge(
    player_minutes_summary[["player_id", "player_name", "primary_position"]], 
    on="player_id", how="left"
).to_csv(output_minutes, index=False)
print(f"✓ Minutos por partido: {output_minutes}")

# 7. Minutos por jugador y posición
output_pos_minutes = os.path.join(OUTPUT_DIR, f"player_minutes_by_position_{season}.csv")
position_minutes.merge(
    player_minutes_summary[["player_id", "player_name", "total_minutes"]], 
    on="player_id", how="left"
).sort_values(["player_id", "minutes_in_position"], ascending=[True, False]).to_csv(output_pos_minutes, index=False)
print(f"✓ Minutos por posición: {output_pos_minutes}")

# 8. Detalle de turnovers (si está habilitado)
if ENABLE_TURNOVER_ANALYSIS and 'turnovers_df' in locals() and not turnovers_df.empty:
    output_turnovers = os.path.join(OUTPUT_DIR, f"turnovers_detail_{season}.csv")
    turnovers_df.to_csv(output_turnovers, index=False)
    print(f"✓ Detalle de turnovers: {output_turnovers}")
    
    # 9. Solo métricas de turnovers per90
    if 'turnover_per90_cols' in locals() and turnover_per90_cols:
        output_turnovers_per90 = os.path.join(OUTPUT_DIR, f"all_players_turnovers_per90_{season}.csv")
        final_df[base_cols + turnover_per90_cols].to_csv(output_turnovers_per90, index=False)
        print(f"✓ Solo turnovers per90: {output_turnovers_per90}")

# 10. Log del análisis
print(f"✓ Log del análisis: {log_filename}")

# ============= 11) RESUMEN FINAL =============
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
    for metric in percentage_metrics:
        if metric in final_df.columns:
            avg = final_df[metric].mean()
            if not np.isnan(avg):
                print(f"  - {metric}: promedio = {avg:.1%}")

if composite_metrics:
    print(f"\n🔢 Métricas compuestas calculadas: {len(composite_metrics)}")
    for metric in composite_metrics:
        if metric in final_df.columns:
            print(f"  - {metric}")

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
print(f"\n✅ ANÁLISIS COMPLETADO EXITOSAMENTE")
print(f"Fin: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print(f"Log guardado en: {log_filename}")
print("="*70)

# Cerrar el logger
sys.stdout.close()
sys.stdout = sys.stdout.terminal  # Restaurar stdout original
