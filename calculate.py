# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import json, ast
import chardet


# ============= CONFIG =============
PATH = "data/events_2024_2025.csv"
season = PATH.split('_', 1)[1].replace('.csv', '')

ASSUME_END_CAP = 120  # límite superior por si hay prórroga
OUTPUT_DIR = "./outputs"

# ============= COLUMNAS A DISCRIMINAR =============
# Lista de columnas por las cuales quieres separar las métricas
COLUMNS_TO_DISCRIMINATE = [
    "type",           # Tipo de evento (Pass, Shot, etc.)
    "pass_height",    # Altura del pase
    "pass_type",
    "pass_switch",
    "duel_type",
    "shot_type",      # Tipo de tiro
    "play_pattern",   # Patrón de juego
    # Agrega más columnas aquí si quieres
]

# Métricas a discriminar (None = todas las numéricas)
METRICS_TO_SPLIT = ["obv_total_net", "shot_statsbomb_xg"]  # O lista: ["obv_for_net", "xg", "xa"]
0
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

# ============= LECTURA =============
print("Cargando datos...")
df = pd.read_csv(PATH, low_memory=False)

df["season"] = season

# df = pd.concat([df1, df2], ignore_index=True)

df["minute"] = pd.to_numeric(df.get("minute", np.nan), errors="coerce")
df["type"] = df["type"].astype(str)

df["_player_id"] = get_player_id_series(df)
df["_player_name"] = get_player_name_series(df)

print(f"Total de eventos cargados: {len(df):,}")

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

# ============= 2) CALCULAR MINUTOS POR PARTIDO =============
print("\nCalculando minutos jugados por partido...")

minutes_rows = []
match_count = 0

for mid, g in df.groupby("match_id", sort=False):
    match_count += 1
    if match_count % 100 == 0:
        print(f"  Procesando partido {match_count}...")
    
    xi_players = lineup_players_from_match(g)
    season = g["season"].iloc[0] if "season" in g.columns else None
    
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
            position = xi_players[pid]['position'] if pid in xi_players else None
            team = xi_players[pid]['team'] if pid in xi_players else None
            
            if not team and "team" in g.columns:
                player_events = g.loc[get_player_id_series(g) == pid]
                if not player_events.empty:
                    team = get_team_name(player_events["team"].iloc[0])
            
            minutes_rows.append({
                "player_id": int(pid),
                "match_id": mid,
                "season": season,
                "minutes": int(mins),
                "starter": is_starter,
                "position": position,
                "team": team
            })

minutes_df = pd.DataFrame(minutes_rows)
print(f"Total de apariciones jugador-partido: {len(minutes_df):,}")

# ============= 3) RESUMEN DE MINUTOS =============
print("\nGenerando resumen de minutos...")

player_minutes_summary = minutes_df.groupby("player_id").agg({
    "minutes": "sum",
    "match_id": "nunique",
    "starter": "sum",
    "season": lambda x: list(x.unique())
}).reset_index()

player_minutes_summary.columns = ["player_id", "total_minutes", "matches_played", "times_starter", "seasons"]

player_minutes_summary["player_name"] = player_minutes_summary["player_id"].map(
    lambda pid: player_info.get(pid, {}).get('name', f"player_{pid}")
)
player_minutes_summary["positions"] = player_minutes_summary["player_id"].map(
    lambda pid: ", ".join(sorted(player_info.get(pid, {}).get('positions', set()))) or None
)
player_minutes_summary["teams"] = player_minutes_summary["player_id"].map(
    lambda pid: ", ".join(sorted(player_info.get(pid, {}).get('teams', set()))) or None
)

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
    "possession", "period", "second", "index", "id", "minute"
}
num_cols = [c for c in num_cols if c not in drop_cols]

print(f"Métricas numéricas a calcular: {len(num_cols)}")

# ============= 5) PREPARAR PARA DISCRIMINACIÓN =============
df["pid"] = get_player_id_series(df).astype("Int64")
all_player_ids = set(player_info.keys())
df_all = df[df["pid"].isin(all_player_ids)].copy()

# Determinar métricas a discriminar
if METRICS_TO_SPLIT is None:
    metrics_to_discriminate = num_cols.copy()
else:
    metrics_to_discriminate = [c for c in METRICS_TO_SPLIT if c in num_cols]

print(f"\nMétricas a discriminar: {len(metrics_to_discriminate)}")

# ============= 5A) DISCRIMINAR POR CADA COLUMNA =============
print("\n" + "="*70)
print("CREANDO MÉTRICAS DISCRIMINADAS")
print("="*70)

all_discriminated_dfs = {}
discrimination_stats = []

for disc_column in COLUMNS_TO_DISCRIMINATE:
    print(f"\n--- Discriminando por: {disc_column} ---")
    
    # Verificar que la columna existe
    if disc_column not in df_all.columns:
        print(f"  ⚠️ Columna '{disc_column}' no encontrada, saltando...")
        continue
    
    # Extraer valores únicos (procesando dicts si es necesario)
    unique_values = df_all[disc_column].apply(get_value_from_column).dropna().unique()
    unique_values = [v for v in unique_values if v]  # Filtrar None/vacíos
    
    print(f"  Valores únicos encontrados: {len(unique_values)}")
    
    # Procesar cada valor
    for value in unique_values:
        # Filtrar eventos por este valor
        df_value = df_all[
            df_all[disc_column].apply(get_value_from_column) == value
        ].copy()
        
        if len(df_value) == 0:
            continue
        
        # Sumar métricas por jugador
        value_sums = df_value.groupby("pid")[metrics_to_discriminate].sum(min_count=1).reset_index()
        value_sums = value_sums.rename(columns={"pid": "player_id"})
        
        # Crear sufijo: columna_valor (normalizado)
        suffix = f"{disc_column}_{str(value).lower().replace(' ', '_').replace('-', '_')}"
        
        # Renombrar columnas
        rename_dict = {col: f"{col}_{suffix}" for col in metrics_to_discriminate}
        value_sums = value_sums.rename(columns=rename_dict)
        
        # Almacenar
        all_discriminated_dfs[suffix] = value_sums
        
        # Stats
        discrimination_stats.append({
            "column": disc_column,
            "value": value,
            "suffix": suffix,
            "total_events": len(df_value),
            "unique_players": value_sums["player_id"].nunique()
        })
        
        print(f"  ✓ {value}: {len(df_value):,} eventos, {value_sums['player_id'].nunique()} jugadores")

# ============= 5B) SUMAR EVENTOS TOTALES =============
print("\n" + "="*70)
print("AGREGANDO EVENTOS TOTALES")
print("="*70)

player_sums = df_all.groupby("pid")[num_cols].sum(min_count=1).reset_index()
player_sums = player_sums.rename(columns={"pid": "player_id"})

# Merge con minutos (incluir times_starter)
player_sums = player_sums.merge(
    player_minutes_summary[[
        "player_id", "total_minutes", "matches_played", "times_starter",
        "player_name", "positions", "teams"
    ]], 
    on="player_id", 
    how="left"
)

print(f"Jugadores en player_sums: {len(player_sums):,}")

# ============= 5C) MERGE DE DISCRIMINACIONES =============
print("\n" + "="*70)
print("COMBINANDO MÉTRICAS DISCRIMINADAS")
print("="*70)

discriminated_cols = []

for suffix, disc_df in all_discriminated_dfs.items():
    player_sums = player_sums.merge(disc_df, on="player_id", how="left")
    new_cols = [c for c in disc_df.columns if c != "player_id"]
    discriminated_cols.extend(new_cols)

print(f"Total métricas discriminadas: {len(discriminated_cols)}")

# Rellenar NaN con 0
player_sums[discriminated_cols] = player_sums[discriminated_cols].fillna(0)

# ============= 6) NORMALIZAR POR 90 MINUTOS =============
print("\n" + "="*70)
print("NORMALIZANDO POR 90 MINUTOS")
print("="*70)

# Métricas totales
for c in num_cols:
    player_sums[f"{c}_per90"] = np.where(
        player_sums["total_minutes"] > 0, 
        player_sums[c] / player_sums["total_minutes"] * 90.0, 
        np.nan
    )

# Métricas discriminadas
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

# ============= 7) PREPARAR DATAFRAME FINAL =============
base_cols = ["player_id", "player_name", "positions", "teams", "total_minutes", "matches_played", "times_starter"]

final_cols = (base_cols + 
              num_cols + 
              discriminated_cols + 
              total_per90_cols + 
              discriminated_per90_cols)

final_df = player_sums[final_cols].copy()
final_df = final_df.sort_values("total_minutes", ascending=False)

# ============= 8) EXPORTS =============
print("\n" + "="*70)
print("EXPORTANDO RESULTADOS")
print("="*70)

import os
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 1. Completo
output_complete = os.path.join(OUTPUT_DIR, "all_players_complete.csv")
final_df.to_csv(output_complete, index=False)
print(f"✓ Completo: {output_complete}")

# 2. Solo per90 (totales + discriminadas)
output_per90_all = os.path.join(OUTPUT_DIR, "all_players_per90_all.csv")
per90_all_cols = base_cols + total_per90_cols + discriminated_per90_cols
final_df[per90_all_cols].to_csv(output_per90_all, index=False)
print(f"✓ Per90 (todas): {output_per90_all}")

# 3. Solo discriminadas per90
output_per90_disc = os.path.join(OUTPUT_DIR, "all_players_per90_discriminated.csv")
disc_per90_cols = base_cols + discriminated_per90_cols
final_df[disc_per90_cols].to_csv(output_per90_disc, index=False)
print(f"✓ Per90 (solo discriminadas): {output_per90_disc}")

# 4. Estadísticas de discriminaciones
disc_stats_df = pd.DataFrame(discrimination_stats).sort_values("total_events", ascending=False)
output_stats = os.path.join(OUTPUT_DIR, "discrimination_statistics.csv")
disc_stats_df.to_csv(output_stats, index=False)
print(f"✓ Estadísticas: {output_stats}")

# 5. Minutos por partido
output_minutes = os.path.join(OUTPUT_DIR, "player_minutes_by_match.csv")
minutes_df.merge(
    player_minutes_summary[["player_id", "player_name"]], 
    on="player_id", how="left"
).to_csv(output_minutes, index=False)
print(f"✓ Minutos: {output_minutes}")

# ============= 9) RESUMEN FINAL =============
print("\n" + "="*70)
print("RESUMEN FINAL")
print("="*70)
print(f"Jugadores procesados: {len(final_df):,}")
print(f"Partidos únicos: {df['match_id'].nunique():,}")
print(f"Total eventos: {len(df):,}")
print(f"Métricas base: {len(num_cols)}")
print(f"Columnas discriminadas: {len(COLUMNS_TO_DISCRIMINATE)}")
print(f"Métricas discriminadas generadas: {len(discriminated_cols)}")
print(f"Total columnas finales: {len(final_df.columns):,}")

print(f"\nTop 15 discriminaciones por eventos:")
print(disc_stats_df.head(15)[["column", "value", "total_events", "unique_players"]].to_string(index=False))

print(f"\nPrimeras 20 columnas discriminadas generadas:")
for col in discriminated_per90_cols[:20]:
    print(f"  - {col}")

print("="*70)