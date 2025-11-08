# -*- coding: utf-8 -*-
"""
Created on Tue Nov  4 19:07:45 2025

@author: guz_m
"""
# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import json, ast

# ====== CONFIG ======
PATH_2425 = r"C:\Users\guz_m\OneDrive\Escritorio\Guz\Football data agency\Monterrey\Proyecto renovaciones\Sergio Ramos\datos_ligamx\events_2024_2025.csv"
PATH_2526 = r"C:\Users\guz_m\OneDrive\Escritorio\Guz\Football data agency\Monterrey\Proyecto renovaciones\Sergio Ramos\datos_ligamx\events_2025_2026.csv"

TARGET_NAME = "Sergio Ramos García"
TARGET_ID   = 5201
ASSUME_MATCH_LENGTH = 90   # si no hay sub off y no queremos usar el último evento

# ====== HELPERS ======
def _coerce_literal(x):
    """Parsea strings con pinta de JSON/dict/list a objeto Python; si falla, devuelve x."""
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

def _norm(s):
    return str(s).strip().lower()

def extract_player_name_ser(df: pd.DataFrame) -> pd.Series:
    """Devuelve serie con nombres (normalizados) desde 'player' o variantes."""
    if "player" in df.columns:
        obj = df["player"].apply(_coerce_literal)
        return obj.apply(lambda v: _norm(v.get("name")) if isinstance(v, dict) else _norm(v))
    for cand in ["player.name", "player_name", "playerName"]:
        if cand in df.columns:
            return df[cand].astype(str).map(_norm)
    return pd.Series([""] * len(df), index=df.index)

def extract_player_id_ser(df: pd.DataFrame) -> pd.Series:
    """Devuelve serie con player_id (si existe) desde columna plana o dentro de 'player'."""
    for cand in ["player_id", "playerId", "idPlayer"]:
        if cand in df.columns:
            return pd.to_numeric(df[cand], errors="coerce")
    if "player" in df.columns:
        obj = df["player"].apply(_coerce_literal)
        return obj.apply(lambda v: v.get("id") if isinstance(v, dict) else np.nan)
    return pd.Series([np.nan] * len(df), index=df.index)

def in_starting_xi(df_match: pd.DataFrame, name_norm_target: str, id_target: int) -> bool:
    """¿Fue titular? Mira eventos 'Starting XI' y lineup dentro de 'tactics'."""
    # 1) filas type == Starting XI
    mask_xi = (df_match["type"].astype(str) == "Starting XI")
    if mask_xi.any():
        # a) por 'player' plano
        names = extract_player_name_ser(df_match.loc[mask_xi])
        ids   = extract_player_id_ser(df_match.loc[mask_xi])
        if (ids == id_target).any() or (names == name_norm_target).any():
            return True
        # b) por 'tactics' -> 'lineup'
        if "tactics" in df_match.columns:
            for tac in df_match.loc[mask_xi, "tactics"].dropna().tolist():
                tac_obj = _coerce_literal(tac)
                if isinstance(tac_obj, dict) and "lineup" in tac_obj:
                    for p in tac_obj.get("lineup", []):
                        try:
                            if p["player"].get("id") == id_target or _norm(p["player"]["name"]) == name_norm_target:
                                return True
                        except Exception:
                            continue
    return False

def minute_sub_on(df_match: pd.DataFrame, name_norm_target: str, id_target: int):
    """Minuto en que ENTRA: Substitution con substitution_replacement = jugador."""
    mask_sub = (df_match["type"].astype(str) == "Substitution")
    if not mask_sub.any() or "substitution_replacement" not in df_match.columns:
        return None
    mins = []
    for _, row in df_match.loc[mask_sub].iterrows():
        repl = _coerce_literal(row["substitution_replacement"])
        minute = pd.to_numeric(row.get("minute", np.nan), errors="coerce")
        pid = np.nan
        pname = ""
        if isinstance(repl, dict):
            pid = repl.get("id", np.nan)
            pname = _norm(repl.get("name"))
        else:
            pname = _norm(str(repl))
        if (pid == id_target) or (pname == name_norm_target):
            if pd.notna(minute):
                mins.append(float(minute))
    return min(mins) if mins else None

def minute_sub_off(df_match: pd.DataFrame, name_norm_target: str, id_target: int):
    """
    Minuto en que SALE: fila Substitution donde 'player' es el jugador que sale.
    (Arreglado: ahora opera SIEMPRE dentro del subset de sustituciones, para evitar
    desalineación de índices con df_match completo).
    """
    if "type" not in df_match.columns:
        return None

    # asegurar tipo string
    types = df_match["type"].astype(str)
    mask_sub = (types == "Substitution")
    if not mask_sub.any():
        return None

    # trabajar SOLO con el subset de sustituciones
    sub_df = df_match.loc[mask_sub].copy()

    # extraer nombres/ids DENTRO del subset
    names = extract_player_name_ser(sub_df)
    ids   = extract_player_id_ser(sub_df)

    # máscara dentro del subset (no sobre df_match completo)
    mask = (ids == id_target) | (names == name_norm_target)

    if mask.any():
        mins = pd.to_numeric(sub_df.loc[mask, "minute"], errors="coerce").dropna()
        if not mins.empty:
            return float(mins.min())
    return None

def match_end_minute(df_match: pd.DataFrame) -> float:
    """Usa el último minuto observado como proxy del fin de partido (mínimo 90)."""
    mx = pd.to_numeric(df_match.get("minute", pd.Series([], dtype=float)), errors="coerce").max()
    if pd.isna(mx):
        return float(ASSUME_MATCH_LENGTH)
    return float(max(ASSUME_MATCH_LENGTH, mx))

def minutes_played(df_match: pd.DataFrame, name_norm_target: str, id_target: int) -> tuple[bool, int]:
    """Calcula (titular, minutos_jugados) con reglas claras."""
    titular = in_starting_xi(df_match, name_norm_target, id_target)
    on = minute_sub_on(df_match, name_norm_target, id_target)
    off = minute_sub_off(df_match, name_norm_target, id_target)

    if titular:
        if off is not None:
            mins = max(0.0, off)  # jugó desde 0 hasta off
        else:
            mins = match_end_minute(df_match)  # 90 o último evento
    else:
        if on is not None and off is None:
            mins = max(0.0, match_end_minute(df_match) - on)
        elif on is not None and off is not None:
            mins = max(0.0, off - on)
        else:
            # si no detectamos nada, ver si tiene eventos -> asumir jugó todo (conservador) o 0
            names = extract_player_name_ser(df_match)
            ids   = extract_player_id_ser(df_match)
            has_evts = ((ids == id_target) | (names == name_norm_target)).any()
            mins = match_end_minute(df_match) if has_evts else 0.0

    return titular, int(round(mins))

# ====== LECTURA ======
df_2425 = pd.read_csv(PATH_2425, low_memory=False)
df_2526 = pd.read_csv(PATH_2526, low_memory=False)
df = pd.concat([df_2425, df_2526], ignore_index=True)

# temporada (best-effort desde el path)
def infer_season_from_path(p):
    s = str(p)
    if "2024" in s and "2025" in s: return "2024-2025"
    if "2025" in s and "2026" in s: return "2025-2026"
    return "2024-2026"

df_2425["_season"] = infer_season_from_path(PATH_2425)
df_2526["_season"] = infer_season_from_path(PATH_2526)
df = pd.concat([df_2425, df_2526], ignore_index=True)

# tipos
if "type" not in df.columns or "match_id" not in df.columns or "minute" not in df.columns:
    raise KeyError("Faltan columnas requeridas: se esperan 'type', 'match_id' y 'minute'.")

df["type"] = df["type"].astype(str)
df["minute"] = pd.to_numeric(df["minute"], errors="coerce")

# ====== LOOP POR PARTIDO ======
rows = []
for mid, g in df.groupby("match_id", sort=False):
    titular, mins = minutes_played(g, _norm(TARGET_NAME), TARGET_ID)

    # equipo (si hay alguna fila del jugador con team)
    team_val = None
    names_g = extract_player_name_ser(g)
    ids_g   = extract_player_id_ser(g)
    mask_player = (ids_g == TARGET_ID) | (names_g == _norm(TARGET_NAME))
    if mask_player.any() and "team" in g.columns:
        tcol = g.loc[mask_player, "team"].dropna()
        if not tcol.empty:
            team_val = tcol.iloc[0]

    # temporada (si alguna de las filas trae _season)
    season_val = g.get("_season", pd.Series([np.nan]*len(g))).dropna()
    season_val = season_val.iloc[0] if not season_val.empty else None

    rows.append({
        "match_id": mid,
        "temporada": season_val,
        "equipo": team_val,
        "titular": titular,
        "minutos_jugados": mins
    })

resumen_df = pd.DataFrame(rows, columns=["match_id","temporada","equipo","titular","minutos_jugados"])
resumen_df = resumen_df[resumen_df["minutos_jugados"] > 0].copy()

# ====== SALIDAS ======
total_partidos = int(len(resumen_df))
total_minutos  = int(resumen_df["minutos_jugados"].sum()) if not resumen_df.empty else 0

print("=== Sergio Ramos — Resumen 24/25 + 25/26 ===")
print(f"Partidos jugados: {total_partidos}")
print(f"Minutos totales: {total_minutos}\n")

if resumen_df.empty:
    print("No se encontraron partidos con minutos > 0.")
else:
    print("Detalle por partido:")
    print(resumen_df.sort_values(["temporada","match_id"]).to_string(index=False))

    # Resumen por temporada
    resumen_temp = (resumen_df
                    .groupby("temporada", dropna=False)
                    .agg(partidos=("match_id","nunique"),
                         minutos=("minutos_jugados","sum"),
                         titularidades=("titular","sum"))
                    .reset_index())
    print("\nResumen por temporada:")
    print(resumen_temp.to_string(index=False))
    
#%%

import pandas as pd
import numpy as np
import json, ast

# ===== CONFIG =====
PATH_2425 = r"C:\Users\guz_m\OneDrive\Escritorio\Guz\Football data agency\Monterrey\Proyecto renovaciones\Sergio Ramos\datos_ligamx\events_2024_2025.csv"
PATH_2526 = r"C:\Users\guz_m\OneDrive\Escritorio\Guz\Football data agency\Monterrey\Proyecto renovaciones\Sergio Ramos\datos_ligamx\events_2025_2026.csv"

TARGET_ID = 5201
TARGET_NAME = "Sergio Ramos García"

CENTER_POSITIONS = ["Center Back", "Left Center Back", "Right Center Back"]
MINUTES_THRESHOLD = 450
ASSUME_MATCH_LENGTH = 90


# ===== HELPERS =====
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


# ===== LECTURA =====
df_2425 = pd.read_csv(PATH_2425, low_memory=False)
df_2526 = pd.read_csv(PATH_2526, low_memory=False)
df = pd.concat([df_2425, df_2526], ignore_index=True)

# ===== LIMPIEZA BASE =====
df["player_id"] = pd.to_numeric(df["player_id"], errors="coerce")
df["minute"] = pd.to_numeric(df["minute"], errors="coerce")
df["period"] = pd.to_numeric(df["period"], errors="coerce")
df["type"] = df["type"].astype(str)
df = df[df["player_id"].notna()]

# Convertir booleanos explícitos (True/False strings u objetos)
bool_cols = [c for c in df.columns if df[c].dropna().isin([True, False, "True", "False"]).any()]
for c in bool_cols:
    df[c] = df[c].astype(str).str.lower().map({"true": 1, "false": 0})
    df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0).astype(int)

# ===== CALCULAR MINUTOS POR JUGADOR =====
# Usamos la ventana [minuto primer evento, último evento] como proxy
minutes_by_player = (
    df.groupby("player_id")["minute"]
    .agg(["min", "max", "count"])
    .rename(columns={"min": "min_first", "max": "min_last", "count": "events"})
)
minutes_by_player["minutes_est"] = minutes_by_player["min_last"] - minutes_by_player["min_first"]
minutes_by_player["minutes_est"] = minutes_by_player["minutes_est"].clip(lower=0, upper=ASSUME_MATCH_LENGTH)
minutes_by_player = minutes_by_player.reset_index()

# ===== SELECCIÓN DE MÉTRICAS =====
num_cols = df.select_dtypes(include=[np.number]).columns.tolist()

# Quitar IDs irrelevantes
drop_cols = ["index", "id", "player_id", "team_id", "possession_team_id", "match_id", "possession"]
num_cols = [c for c in num_cols if c not in drop_cols]

# ===== AGREGADOS POR JUGADOR =====
agg_funcs = {c: "mean" for c in num_cols}
player_stats = df.groupby(["player_id", "position"]).agg(agg_funcs).reset_index()

# Agregar minutos estimados
player_stats = player_stats.merge(minutes_by_player[["player_id", "minutes_est"]], on="player_id", how="left")

# Filtrar por minutos y posición central
centers = player_stats[
    (player_stats["position"].isin(CENTER_POSITIONS)) &
    (player_stats["minutes_est"] > MINUTES_THRESHOLD)
].copy()

# ===== COMPARAR SERGIO RAMOS =====
ramos_row = centers.loc[centers["player_id"] == TARGET_ID]
if ramos_row.empty:
    raise ValueError("⚠️ No se encontraron eventos o posición válida para Sergio Ramos (5201).")

# promedio del grupo
group_mean = centers[num_cols].mean(numeric_only=True)
group_std = centers[num_cols].std(numeric_only=True)

ramos_values = ramos_row[num_cols].iloc[0]
diffs = ramos_values - group_mean
z_scores = (ramos_values - group_mean) / group_std

summary = pd.DataFrame({
    "Métrica": num_cols,
    "Ramos": ramos_values.values,
    "Promedio_CBs": group_mean.values,
    "Diferencia": diffs.values,
    "Z_Score": z_scores.values
}).replace([np.inf, -np.inf], np.nan).dropna(subset=["Z_Score"])

summary = summary.sort_values("Z_Score", ascending=False)

# ===== RESULTADOS =====
print(f"=== Sergio Ramos vs Center Backs (>= {MINUTES_THRESHOLD} min) ===")
print(f"Jugadores comparados: {len(centers)}")
print(f"Métricas evaluadas: {len(num_cols)}\n")

print("🔝 TOP 15 métricas por encima del promedio:")
print(summary.head(15).to_string(index=False))

print("\n🔻 15 métricas por debajo del promedio:")
print(summary.tail(15).to_string(index=False))

# (Opcional) Exportar todo a CSV
summary.to_csv("ramos_vs_centerbacks.csv", index=False)
print("\nResumen completo exportado como 'ramos_vs_centerbacks.csv'")

#%%

## Normalizar datos cada 90 minutos ##

# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import json, ast

# ========= CONFIG =========
PATH_2425 = r"C:\Users\guz_m\OneDrive\Escritorio\Guz\Football data agency\Monterrey\Proyecto renovaciones\Sergio Ramos\datos_ligamx\events_2024_2025.csv"
PATH_2526 = r"C:\Users\guz_m\OneDrive\Escritorio\Guz\Football data agency\Monterrey\Proyecto renovaciones\Sergio Ramos\datos_ligamx\events_2025_2026.csv"

CENTER_POSITIONS = {"Center Back", "Left Center Back", "Right Center Back"}
TARGET_ID = 5201
MINUTES_THRESHOLD = 450
ASSUME_MATCH_END_CAP = 120  # por si hay prórroga o adición larga

# ========= HELPERS =========
def _coerce_literal(x):
    """Convierte str con pinta de JSON/list/dict a objeto Python si es posible."""
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

def _norm(s):
    return str(s).strip().lower()

def extract_player_id_ser(df: pd.DataFrame) -> pd.Series:
    """player_id desde columna plana o dentro de 'player' (dict)."""
    for cand in ["player_id", "playerId", "idPlayer"]:
        if cand in df.columns:
            return pd.to_numeric(df[cand], errors="coerce")
    if "player" in df.columns:
        obj = df["player"].apply(_coerce_literal)
        return obj.apply(lambda v: v.get("id") if isinstance(v, dict) else np.nan)
    return pd.Series(np.nan, index=df.index)

def extract_player_name_ser(df: pd.DataFrame) -> pd.Series:
    """nombre del jugador desde 'player' o variantes."""
    if "player" in df.columns:
        obj = df["player"].apply(_coerce_literal)
        return obj.apply(lambda v: v.get("name") if isinstance(v, dict) else str(v))
    for cand in ["player.name", "player_name", "playerName"]:
        if cand in df.columns:
            return df[cand].astype(str)
    return pd.Series("", index=df.index)

def get_position_str(val):
    """Devuelve posición como string plano (tal cual)."""
    if pd.isna(val):
        return None
    v = _coerce_literal(val)
    if isinstance(v, dict):
        return v.get("name") or v.get("label")
    return str(v)

def match_end_minute(df_match: pd.DataFrame) -> float:
    """Fin de partido como máximo minuto observado (cap en 120)."""
    mx = pd.to_numeric(df_match.get("minute", pd.Series([], dtype=float)), errors="coerce").max()
    if pd.isna(mx):
        return 90.0
    return float(min(ASSUME_MATCH_END_CAP, max(90.0, mx)))

def in_starting_xi(df_match: pd.DataFrame, player_id: int) -> bool:
    """¿Está el jugador en el XI inicial? Se mira tactics.lineup en filas type=Starting XI."""
    types = df_match["type"].astype(str)
    mask_xi = (types == "Starting XI")
    if not mask_xi.any() or "tactics" not in df_match.columns:
        return False
    for tac in df_match.loc[mask_xi, "tactics"].dropna().tolist():
        tac_obj = _coerce_literal(tac)
        if isinstance(tac_obj, dict) and "lineup" in tac_obj:
            for p in tac_obj.get("lineup", []):
                pid = (p.get("player") or {}).get("id")
                if pid == player_id:
                    return True
    return False

def minute_sub_on(df_match: pd.DataFrame, player_id: int) -> float | None:
    """Minuto en que ENTRA: fila Substitution con substitution_replacement = jugador."""
    types = df_match["type"].astype(str)
    sub_df = df_match.loc[types == "Substitution"].copy()
    if sub_df.empty or "substitution_replacement" not in sub_df.columns:
        return None
    mins = []
    for _, row in sub_df.iterrows():
        repl = _coerce_literal(row["substitution_replacement"])
        if isinstance(repl, dict) and repl.get("id") == player_id:
            m = pd.to_numeric(row.get("minute", np.nan), errors="coerce")
            if pd.notna(m):
                mins.append(float(m))
    return min(mins) if mins else None

def minute_sub_off(df_match: pd.DataFrame, player_id: int) -> float | None:
    """Minuto en que SALE: fila Substitution donde 'player' es el que sale."""
    types = df_match["type"].astype(str)
    sub_df = df_match.loc[types == "Substitution"].copy()
    if sub_df.empty:
        return None
    # 'player' puede ser dict o string; usamos extract_player_id_ser sobre el subset
    ids = extract_player_id_ser(sub_df)
    mask = (ids == player_id)
    if mask.any():
        mins = pd.to_numeric(sub_df.loc[mask, "minute"], errors="coerce").dropna()
        if not mins.empty:
            return float(mins.min())
    return None

def minutes_played_one_match(df_match: pd.DataFrame, player_id: int) -> int:
    """
    Minutos jugados por un player_id en un match: titulares/subs con Substitution.
    """
    end_min = match_end_minute(df_match)
    started = in_starting_xi(df_match, player_id)
    on = minute_sub_on(df_match, player_id)
    off = minute_sub_off(df_match, player_id)

    if started:
        if off is not None:
            return int(round(max(0.0, off)))
        return int(round(end_min))
    else:
        if on is not None and off is None:
            return int(round(max(0.0, end_min - on)))
        if on is not None and off is not None:
            return int(round(max(0.0, off - on)))
    # no aparece ni en XI ni en subs => 0
    return 0

# ========= LECTURA =========
df1 = pd.read_csv(PATH_2425, low_memory=False)
df2 = pd.read_csv(PATH_2526, low_memory=False)
df = pd.concat([df1, df2], ignore_index=True)

# Tipos base
df["minute"] = pd.to_numeric(df.get("minute", np.nan), errors="coerce")
df["type"] = df["type"].astype(str)

# ========= 1) JUGADORES CENTRALES CANDIDATOS (por position en eventos) =========
# Usamos exactamente los strings que indicaste
if "position" not in df.columns:
    raise KeyError("No existe la columna 'position' en los CSV.")

pos_series = df["position"].apply(get_position_str)
mask_cb_evt = pos_series.isin(CENTER_POSITIONS)

# Obtenemos (player_id, player_name) de esos eventos
df["player_id_extracted"] = extract_player_id_ser(df)
df["player_name_extracted"] = extract_player_name_ser(df)

cb_candidates = (
    df.loc[mask_cb_evt & df["player_id_extracted"].notna(), ["player_id_extracted", "player_name_extracted"]]
    .drop_duplicates()
    .rename(columns={"player_id_extracted": "player_id", "player_name_extracted": "player_name"})
)

# Asegurar que Ramos esté en la lista (por si justo no tuvo un evento con 'position' seteada en algún partido)
if not (cb_candidates["player_id"] == TARGET_ID).any():
    # si aparece en algún evento, lo sumamos igual
    any_ramos = df.loc[extract_player_id_ser(df) == TARGET_ID, ["player_id_extracted","player_name_extracted"]].dropna()
    if not any_ramos.empty:
        rrow = any_ramos.iloc[0]
        cb_candidates = pd.concat([
            cb_candidates,
            pd.DataFrame([{"player_id": int(rrow["player_id_extracted"]), "player_name": rrow["player_name_extracted"]}])
        ], ignore_index=True).drop_duplicates(subset=["player_id"])

# ========= 2) MINUTOS POR JUGADOR (sumando partidos que jugaron) =========
# Para cada partido, construimos conjunto de participantes: eventos + subs (sale/entra) + XI
minutes_rows = []
for mid, g in df.groupby("match_id", sort=False):
    # participantes por eventos
    participants = set(extract_player_id_ser(g).dropna().astype(int).tolist())

    # + los que salen en Substitution
    sub_df = g.loc[g["type"] == "Substitution"]
    if not sub_df.empty:
        participants.update(extract_player_id_ser(sub_df).dropna().astype(int).tolist())
        # + los que entran
        if "substitution_replacement" in sub_df.columns:
            for repl in sub_df["substitution_replacement"].dropna().tolist():
                obj = _coerce_literal(repl)
                if isinstance(obj, dict):
                    pid = obj.get("id")
                    if pid is not None:
                        participants.add(int(pid))

    # + XI desde tactics.lineup
    xi_df = g.loc[g["type"] == "Starting XI"]
    if not xi_df.empty and "tactics" in xi_df.columns:
        for tac in xi_df["tactics"].dropna().tolist():
            tac_obj = _coerce_literal(tac)
            if isinstance(tac_obj, dict):
                for p in tac_obj.get("lineup", []):
                    pid = (p.get("player") or {}).get("id")
                    if pid is not None:
                        participants.add(int(pid))

    # calcular minutos por jugador en este match
    for pid in participants:
        mins = minutes_played_one_match(g, pid)
        if mins > 0:
            minutes_rows.append({"match_id": mid, "player_id": int(pid), "minutes": int(mins)})

minutes_df = pd.DataFrame(minutes_rows)
minutes_total = minutes_df.groupby("player_id", as_index=False)["minutes"].sum()

# ========= 3) FILTRAR A CENTRALES CON >= 450' =========
players_minutes = cb_candidates.merge(minutes_total, on="player_id", how="left")
players_minutes["minutes"] = players_minutes["minutes"].fillna(0).astype(int)
cb_pool = players_minutes.loc[players_minutes["minutes"] >= MINUTES_THRESHOLD].copy()

if cb_pool.empty:
    raise ValueError("No se encontraron centrales con minutos >= 450. Revisá si los XI/substitutions están presentes.")

# ========= 4) PREPARAR MÉTRICAS: booleanos -> 0/1, numéricas por 90' =========
# Convertir booleanos (True/False o 'true'/'false') a 0/1
bool_cols = []
for c in df.columns:
    vals = df[c].dropna().astype(str).str.lower()
    if not vals.empty and vals.isin(["true", "false"]).all():
        bool_cols.append(c)

for c in bool_cols:
    df[c] = df[c].astype(str).str.lower().map({"true": 1, "false": 0}).astype("float64")

# columnas numéricas
num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
# quitamos identificadores y tiempo que no queremos sumar como métrica
drop_cols = {
    "player_id", "player_id_extracted", "team_id", "match_id", "possession_team_id",
    "possession", "period", "second", "index", "id", "minute"  # 'minute' no es métrica por90
}
num_cols = [c for c in num_cols if c not in drop_cols]

# Sumas por jugador solo para el pool de centrales (en todo el dataset)
df["pid"] = extract_player_id_ser(df).astype("Int64")
pool_ids = set(cb_pool["player_id"].astype(int).tolist())
df_pool = df[df["pid"].isin(pool_ids)].copy()

player_sums = df_pool.groupby("pid")[num_cols].sum(min_count=1).reset_index().rename(columns={"pid": "player_id"})
player_sums = player_sums.merge(cb_pool[["player_id","minutes","player_name"]], on="player_id", how="left")

# Por 90'
for c in num_cols:
    player_sums[c] = np.where(player_sums["minutes"] > 0, player_sums[c] / player_sums["minutes"] * 90.0, np.nan)

# ========= 5) COMPARACIÓN RAMOS VS RESTO =========
if not (player_sums["player_id"] == TARGET_ID).any():
    raise ValueError("Sergio Ramos (5201) no quedó en el pool >=450'. Verificar minutos calculados.")

ramos_row = player_sums.loc[player_sums["player_id"] == TARGET_ID]
others = player_sums.loc[player_sums["player_id"] != TARGET_ID]

group_mean = others[num_cols].mean(numeric_only=True)
group_std  = others[num_cols].std(numeric_only=True)

ramos_vals = ramos_row[num_cols].iloc[0]
diffs = ramos_vals - group_mean
zscore = (ramos_vals - group_mean) / group_std

summary = pd.DataFrame({
    "Métrica": num_cols,
    "Ramos_por90": ramos_vals.values,
    "Promedio_CBs_por90": group_mean.values,
    "Diferencia": diffs.values,
    "Z_Score": zscore.values
}).replace([np.inf, -np.inf], np.nan).dropna(subset=["Z_Score"]).sort_values("Z_Score", ascending=False)

# ========= 6) SALIDAS =========
print(f"=== Centrales (CB/LCB/RCB) >= {MINUTES_THRESHOLD} min — por 90' ===")
print(f"Jugadores en el pool: {len(cb_pool)}  |  Otros (para promedio): {len(others)}")
print("\n🔝 TOP 15 métricas por encima del promedio (Ramos):")
print(summary.head(15).to_string(index=False))

print("\n🔻 15 métricas por debajo del promedio (Ramos):")
print(summary.tail(15).to_string(index=False))

# Exports opcionales
summary.to_csv("ramos_vs_centrales_por90.csv", index=False)
cb_pool.sort_values("minutes", ascending=False).to_csv("centrales_minutos.csv", index=False)
print("\nCSV exportados: ramos_vs_centrales_por90.csv, centrales_minutos.csv")


#%%

## Segundo intento ##

# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import json, ast
from collections import defaultdict, Counter

# ============= CONFIG =============
PATH_2425 = r"C:\Users\guz_m\OneDrive\Escritorio\Guz\Football data agency\Monterrey\Proyecto renovaciones\Sergio Ramos\datos_ligamx\events_2024_2025.csv"
PATH_2526 = r"C:\Users\guz_m\OneDrive\Escritorio\Guz\Football data agency\Monterrey\Proyecto renovaciones\Sergio Ramos\datos_ligamx\events_2025_2026.csv"

TARGET_ID = 5201
CENTER_POSITIONS = {"Center Back", "Left Center Back", "Right Center Back"}
MINUTES_THRESHOLD = 450
ASSUME_END_CAP = 120  # límite superior por si hay prórroga

# ============= HELPERS =============
def _coerce_literal(x):
    """Parsea str con pinta de JSON/dict/list, si no, devuelve x."""
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

def match_end_minute(df_match):
    """Final del partido por minuto máximo observado (cap a ASSUME_END_CAP; mínimo 90)."""
    mx = pd.to_numeric(df_match.get("minute", pd.Series([], dtype=float)), errors="coerce").max()
    if pd.isna(mx):
        return 90.0
    return float(min(ASSUME_END_CAP, max(90.0, mx)))

def lineup_players_from_match(df_match):
    """
    Extrae XI titular del partido desde eventos type='Starting XI' -> tactics.lineup.
    Devuelve dict: {player_id: position_name}
    """
    out = {}
    xi = df_match.loc[df_match["type"].astype(str) == "Starting XI"]
    if xi.empty or "tactics" not in xi.columns:
        return out
    for tac in xi["tactics"].dropna().tolist():
        tac_obj = _coerce_literal(tac)
        if isinstance(tac_obj, dict):
            for p in tac_obj.get("lineup", []):
                pid = (p.get("player") or {}).get("id")
                pos = (p.get("position") or {}).get("name") or p.get("position")
                if pid is not None and pos:
                    out[int(pid)] = str(pos)
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
df1 = pd.read_csv(PATH_2425, low_memory=False)
df2 = pd.read_csv(PATH_2526, low_memory=False)
df = pd.concat([df1, df2], ignore_index=True)

# Tipos base
df["minute"] = pd.to_numeric(df.get("minute", np.nan), errors="coerce")
df["type"] = df["type"].astype(str)

# Player fields auxiliares
df["_player_id"] = get_player_id_series(df)
df["_player_name"] = get_player_name_series(df)

# ============= 1) CANDIDATOS CB DESDE STARTING XI (oficial) + fallback por eventos =============
# a) XI oficial -> todos los (player_id, position) de cada partido
cb_candidates = set()
name_by_id = {}

for mid, g in df.groupby("match_id", sort=False):
    xi_players = lineup_players_from_match(g)  # {pid: pos_name}
    # registrar nombres si están disponibles
    if not xi_players:
        continue
    # registrar candidatos CB por posiciones exactas
    for pid, pos in xi_players.items():
        if pos in CENTER_POSITIONS:
            cb_candidates.add(int(pid))
    # asociar nombre si aparece en el match
    ids = get_player_id_series(g)
    names = get_player_name_series(g)
    tmp = pd.DataFrame({"pid": ids, "pname": names}).dropna()
    for _, row in tmp.iterrows():
        try:
            name_by_id[int(row["pid"])] = str(row["pname"])
        except Exception:
            pass

# b) Fallback: si no hubo XI en algún partido, detectar por position.name de eventos
if "position" in df.columns:
    pos_evt = df["position"].apply(get_position_str)
    mask_cb_evt = pos_evt.isin(CENTER_POSITIONS)
    ids_evt = df.loc[mask_cb_evt, "_player_id"].dropna().astype(int).unique().tolist()
    cb_candidates.update(ids_evt)

# Forzar a Ramos a la lista (por seguridad)
if TARGET_ID not in cb_candidates and df["_player_id"].eq(TARGET_ID).any():
    cb_candidates.add(TARGET_ID)

# ============= 2) MINUTOS POR PLAYER (sumando partidos) =============
minutes_rows = []
for mid, g in df.groupby("match_id", sort=False):
    xi_players = lineup_players_from_match(g)  # dict de XI (para titulares reales)
    # Participantes: XI + subs (sale/entra) + cualquiera con evento
    participants = set(xi_players.keys())
    subs = g.loc[g["type"] == "Substitution"]
    if not subs.empty:
        # sale
        ids_out = get_player_id_series(subs).dropna().astype(int).tolist()
        participants.update(ids_out)
        # entra
        if "substitution_replacement" in subs.columns:
            for repl in subs["substitution_replacement"].dropna().tolist():
                obj = _coerce_literal(repl)
                if isinstance(obj, dict) and obj.get("id") is not None:
                    participants.add(int(obj["id"]))
    # cualquiera con evento
    participants.update(get_player_id_series(g).dropna().astype(int).tolist())

    # calcular minutos solo para quienes estén en el pool candidato CB (o Ramos)
    for pid in participants:
        if (pid in cb_candidates) or (pid == TARGET_ID):
            mins = minutes_played_in_match(g, int(pid), xi_players)
            if mins > 0:
                # registrar nombre si aparece
                if pid not in name_by_id:
                    subset = g.loc[get_player_id_series(g) == pid]
                    if not subset.empty:
                        nm = get_player_name_series(subset).dropna()
                        if not nm.empty:
                            name_by_id[pid] = str(nm.iloc[0])
                minutes_rows.append({"player_id": int(pid), "match_id": mid, "minutes": int(mins)})

minutes_df = pd.DataFrame(minutes_rows)
minutes_total = minutes_df.groupby("player_id", as_index=False)["minutes"].sum()

# ============= 3) POOL CBs >= 450' =============
pool = minutes_total.loc[minutes_total["player_id"].isin(cb_candidates)].copy()
pool["minutes"] = pool["minutes"].fillna(0).astype(int)
pool = pool.loc[pool["minutes"] >= MINUTES_THRESHOLD].copy()

if pool.empty:
    raise ValueError("No se encontraron centrales (CB/LCB/RCB) con minutos >= 450.")

# Agregar nombre
pool["player_name"] = pool["player_id"].map(lambda pid: name_by_id.get(int(pid), f"player_{int(pid)}"))

# ============= 4) BOOLEANS -> 0/1 y métricas por 90' =============
# Detectar booleanos (True/False o 'true'/'false')
bool_cols = []
for c in df.columns:
    vals = df[c].dropna().astype(str).str.lower()
    if len(vals) == 0:
        continue
    if vals.isin(["true", "false"]).all():
        bool_cols.append(c)

for c in bool_cols:
    df[c] = df[c].astype(str).str.lower().map({"true": 1, "false": 0}).astype("float64")

# Numéricos
num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
# Remover identificadores y tiempo/minuto que no son métricas por90
drop_cols = {
    "_player_id", "player_id", "team_id", "match_id", "possession_team_id",
    "possession", "period", "second", "index", "id", "minute"
}
num_cols = [c for c in num_cols if c not in drop_cols]

# Sumar solo los eventos de los jugadores del pool
df["pid"] = get_player_id_series(df).astype("Int64")
pool_ids = set(pool["player_id"].astype(int).tolist())
df_pool = df[df["pid"].isin(pool_ids)].copy()

player_sums = df_pool.groupby("pid")[num_cols].sum(min_count=1).reset_index().rename(columns={"pid": "player_id"})
player_sums = player_sums.merge(pool, on="player_id", how="left")  # agrega minutes y name

# Por 90'
for c in num_cols:
    player_sums[c] = np.where(player_sums["minutes"] > 0, player_sums[c] / player_sums["minutes"] * 90.0, np.nan)

# ============= 5) COMPARAR RAMOS VS GRUPO (excluyéndolo) =============
if TARGET_ID not in player_sums["player_id"].values:
    raise ValueError("Sergio Ramos (5201) no alcanzó el umbral de minutos en el pool de CBs.")

ramos_row = player_sums.loc[player_sums["player_id"] == TARGET_ID]
others = player_sums.loc[player_sums["player_id"] != TARGET_ID]

group_mean = others[num_cols].mean(numeric_only=True)
group_std  = others[num_cols].std(numeric_only=True)

r_vals = ramos_row[num_cols].iloc[0]
diffs  = r_vals - group_mean
zscore = (r_vals - group_mean) / group_std

summary = (pd.DataFrame({
    "Métrica": num_cols,
    "Ramos_por90": r_vals.values,
    "Promedio_CBs_por90": group_mean.values,
    "Diferencia": diffs.values,
    "Z_Score": zscore.values
})
.replace([np.inf, -np.inf], np.nan)
.dropna(subset=["Z_Score"])
.sort_values("Z_Score", ascending=False))

# ============= 6) SALIDAS =============
print(f"=== Centrales (CB/LCB/RCB) ≥ {MINUTES_THRESHOLD} min — por 90' (StatsBomb v4) ===")
print(f"Jugadores en el pool: {len(pool)}  |  Otros (para promedio): {len(others)}")
print("\n🔝 TOP 15 métricas por encima del promedio (Ramos):")
print(summary.head(15).to_string(index=False))
print("\n🔻 15 métricas por debajo del promedio (Ramos):")
print(summary.tail(15).to_string(index=False))

# Exports
summary.to_csv("ramos_vs_centrales_por90_v4.csv", index=False)
pool.sort_values("minutes", ascending=False).to_csv("centrales_minutes_pool.csv", index=False)
player_sums.sort_values("minutes", ascending=False).to_csv("centrales_metrics_por90.csv", index=False)

print("\nCSV exportados:")
print(" - ramos_vs_centrales_por90_v4.csv (comparación completa)")
print(" - centrales_minutes_pool.csv (jugadores CB ≥450’)")
print(" - centrales_metrics_por90.csv (todas las métricas por90 del pool)")
