# -*- coding: utf-8 -*-
"""
goalkeeper_metrics_builder.py
Calcula métricas avanzadas de arqueros desde eventos de StatsBomb.

Métricas calculadas (todas prefijadas con 'gk_'):

  Shot Stopping (SIN penalties):
    gk_save_pct                                     % de tiros a puerta detenidos
    gk_psxg_faced_per90                             PSxG/Goals Prevented por 90min*
    gk_goals_prevented                              Goles prevenidos*
    gk_goals_prevented_per90                        Goles prevenidos por 90*
    gk_shots_on_target_against_per90                Tiros a puerta enfrentados por 90
    gk_saves_per90                                  Paradas por 90
    
    *NOTA IMPORTANTE sobre goals_prevented:
      - Si se usa shot_gk_shot_stopping_xg_suppression (columna de StatsBomb):
          → El valor YA ES goals prevented directamente (no se resta nada)
          → gk_psxg_faced = goals_prevented (misma métrica)
      - Si se usan fallbacks (shot_shot_execution_xg o shot_statsbomb_xg):
          → goals_prevented = PSxG - goals_conceded (cálculo tradicional)
          → gk_psxg_faced ≠ goals_prevented

  Goals Conceded (SIN penalties, SIN own goals):
    gk_goals_conceded_total                         Goles concedidos
    gk_goals_conceded_per90                         Goles concedidos por 90
    gk_own_goals_against                            Autogoles (métrica separada)
    gk_own_goals_against_per90                      Autogoles por 90
    gk_errors_leading_to_goal                       Errores del arquero que llevaron a gol

  Penalties (métricas separadas):
    gk_penalty_save_pct                             % de penalties atajados
    gk_penalties_faced                              Penalties enfrentados (total)
    gk_penalties_saved                              Penalties atajados (total)
    gk_penalties_conceded                           Goles de penalty concedidos (total)
    gk_penalties_psxg                               PSxG total de penalties (informativo)
    gk_penalties_faced_per90                        Penalties enfrentados por 90
    gk_penalties_saved_per90                        Penalties atajados por 90
    gk_penalties_conceded_per90                     Goles de penalty concedidos por 90

  Foot Play:
    gk_pass_completion_pct                          % de pases completados
    gk_pass_obv_per90                               OBV generado por pases por 90
    gk_long_ball_pct                                % de pases largos (>32m)
    gk_pressured_passes_def_third_completion_pct    % completados bajo presión en tercio defensivo

  Outside Box:
    gk_actions_outside_box_per90                    Acciones fuera del área por 90
    gk_aggressive_distance_avg                      Distancia promedio de juego (x)

IMPORTANTE - EXCLUSIONES Y SEPARACIÓN:
  ❌ PENALTIES: Excluidos de las métricas de shot stopping principales
     - NO afectan: gk_save_pct, gk_goals_prevented, gk_psxg_faced (principales)
     - SÍ tienen métricas propias: gk_penalties_*, gk_penalty_save_pct
     - Razón: Permite comparar shot stopping en juego abierto vs situaciones de penalty
  
  ❌ OWN GOALS: NO incluidos en goals_conceded_total
     - Tienen métricas propias: gk_own_goals_against, gk_own_goals_against_per90
     - Razón: No son "culpa" del arquero para métricas de rendimiento

  ✅ MÉTRICAS PRINCIPALES INCLUYEN: Open Play, Free Kicks, Corners

COLUMNA PSxG UTILIZADA Y CÁLCULO DE GOALS PREVENTED:
  Prioridad de columnas:
    1. shot_gk_shot_stopping_xg_suppression - Métrica de StatsBomb
       → YA ES goals prevented (suma directa, no restar goals)
    2. shot_shot_execution_xg - Post-shot xG general (fallback)
       → Requiere cálculo: goals_prevented = PSxG - goals_conceded
    3. shot_statsbomb_xg - Pre-shot xG (fallback)
       → Requiere cálculo: goals_prevented = PSxG - goals_conceded

DISEÑO:
  - Se llama DESDE main_analysis.py (no se ejecuta solo)
  - Reutiliza df_all y player_minutes_summary que ya existen en main_analysis
  - El linking GK→Shot se hace por related_events (preciso con sustituciones)
  - Retorna DataFrame con player_id + métricas, listo para merge con final_df
"""

import numpy as np
import pandas as pd
import ast
import json


# ============= CONSTANTES =============
# Outcomes de goalkeeper_outcome que indican error
ERROR_OUTCOMES = {"Touched Out", "Punched Out", "Incomplete", "No Catch", "Lost"}

# Umbral de distancia para pases largos (metros, ~35 yards)
LONG_BALL_M = 32.0

# Tercio defensivo: x < este valor (cancha StatsBomb = 120m)
DEF_THIRD_X = 40.0

# Línea del área de penal: acciones del GK con x > este valor están fuera del área
BOX_LINE_X = 18.0


# ============= HELPERS =============

def _parse_related(val):
    """
    Parse columna related_events → lista de strings (UUIDs).
    
    En el CSV puede ser: "['uuid1', 'uuid2']" (string) o ya una lista.
    Retorna lista vacía si no se puede parsear.
    """
    if pd.isna(val) or val is None:
        return []
    if isinstance(val, list):
        return [str(v) for v in val]
    if isinstance(val, str):
        s = val.strip()
        if s.startswith("["):
            try:
                return [str(v) for v in json.loads(s)]
            except Exception:
                try:
                    return [str(v) for v in ast.literal_eval(s)]
                except Exception:
                    return []
    return []


def _get_pid_col(df):
    """Retorna nombre de la columna de player_id disponible en df."""
    for col in ("_player_id", "pid", "player_id"):
        if col in df.columns:
            return col
    return None


def _get_psxg_col(df):
    """
    Retorna nombre de columna PSxG.
    
    Prioridad:
      1. shot_gk_shot_stopping_xg_suppression - Métrica específica de StatsBomb para shot stopping
      2. shot_shot_execution_xg - Post-shot xG general
      3. shot_statsbomb_xg - Pre-shot xG (fallback)
    """
    if "shot_gk_shot_stopping_xg_suppression" in df.columns:
        return "shot_gk_shot_stopping_xg_suppression"
    if "shot_shot_execution_xg" in df.columns:
        return "shot_shot_execution_xg"
    if "shot_statsbomb_xg" in df.columns:
        return "shot_statsbomb_xg"
    return None


def _identify_gks(df, player_minutes_summary, pid_col):
    """
    Identifica player_ids de arqueros por 3 métodos combinados:
      1. primary_position == 'Goalkeeper' en player_minutes_summary
      2. Eventos con type == 'Goal Keeper'
      3. Eventos con position que contiene 'Goalkeeper'
    
    Usar los 3 métodos hace la identificación robusta ante CSVs con columnas
    faltantes o formatos inconsistentes.
    """
    gk_ids = set()

    # 1) Desde player_minutes_summary (fuente más confiable para posición)
    if "primary_position" in player_minutes_summary.columns:
        mask = player_minutes_summary["primary_position"] == "Goalkeeper"
        gk_ids.update(
            player_minutes_summary.loc[mask, "player_id"].dropna().tolist()
        )

    # 2) Desde eventos type == 'Goal Keeper'
    gk_type_events = df[df["type"] == "Goal Keeper"]
    if not gk_type_events.empty:
        gk_ids.update(gk_type_events[pid_col].dropna().unique().tolist())

    # 3) Desde columna position
    if "position" in df.columns:
        pos_mask = df["position"].astype(str).str.contains(
            "Goalkeeper", case=False, na=False
        )
        if pos_mask.any():
            gk_ids.update(df.loc[pos_mask, pid_col].dropna().unique().tolist())

    return gk_ids


# ============= LINKING GK → SHOTS =============

def _build_gk_shot_links(df, pid_col, psxg_col):
    """
    Construye DataFrame de links GK event → Shot usando related_events.
    
    Proceso:
      1. Filtrar eventos type == 'Goal Keeper'
      2. Parse + explotar related_events → una fila por ID relacionado
      3. Merge con shots por 'id' → solo quedan los related_events que son shots
    
    El resultado tiene una fila por cada par (GK event, Shot relacionado).
    Esto es la clave de precisión: cada shot se atribuye al GK que lo enfrentó,
    no a todos los GKs del partido. Maneja sustituciones correctamente.
    
    Retorna DataFrame con columnas del GK event + _shot_outcome + _psxg.
    """
    # Filtrar GK events
    df_gk = df[df["type"] == "Goal Keeper"].copy()
    if df_gk.empty:
        return pd.DataFrame()

    # Parse y explotar related_events
    df_gk["_related"] = df_gk["related_events"].apply(_parse_related)
    df_gk_exp = df_gk.explode("_related").rename(columns={"_related": "_rel_id"})
    # Filtrar filas sin related event válido
    df_gk_exp = df_gk_exp[
        df_gk_exp["_rel_id"].notna() & (df_gk_exp["_rel_id"].astype(str).str.len() > 0)
    ]

    if df_gk_exp.empty:
        return pd.DataFrame()

    # Preparar shots para merge (solo columnas necesarias)
    df_shots = df[df["type"] == "Shot"].copy()
    if df_shots.empty:
        return pd.DataFrame()
    
    # FILTRAR PENALTIES - excluir completamente de todas las métricas
    df_shots = df_shots[df_shots["shot_type"] != "Penalty"]
    
    if df_shots.empty:
        return pd.DataFrame()

    shots_cols = ["id", "shot_outcome", "shot_type"]
    if psxg_col and psxg_col in df_shots.columns:
        df_shots[psxg_col] = pd.to_numeric(df_shots[psxg_col], errors="coerce").fillna(0)
        shots_cols.append(psxg_col)

    shots_lk = df_shots[shots_cols].copy()
    shots_lk["id"] = shots_lk["id"].astype(str)

    # Rename para merge
    rename_map = {"id": "_rel_id", "shot_outcome": "_shot_outcome", "shot_type": "_shot_type"}
    if psxg_col and psxg_col in shots_lk.columns:
        rename_map[psxg_col] = "_psxg"
    shots_lk = shots_lk.rename(columns=rename_map)

    # Merge: solo los related_events que corresponden a shots quedan (inner join)
    linked = df_gk_exp.merge(shots_lk, on="_rel_id", how="inner")

    return linked


# ============= AGREGACIONES POR CATEGORÍA =============

def _agg_shot_stopping(linked, pid_col, psxg_col):
    """
    Agrega métricas de shot stopping por arquero.
    
    Input: DataFrame de links GK→Shot (output de _build_gk_shot_links).
    
    IMPORTANTE - Cálculo de goals_prevented:
      - Si psxg_col == 'shot_gk_shot_stopping_xg_suppression':
          goals_prevented = suma de PSxG (ya es goals prevented directamente)
      - Si psxg_col es otro (shot_shot_execution_xg o shot_statsbomb_xg):
          goals_prevented = suma de PSxG - goals concedidos
    
    Métricas:
      - shots_on_target: Saved + Goal (los únicos que llegaron al arquero)
      - saves: outcome == 'Saved'
      - save_pct: saves / shots_on_target
      - goals_conceded_shots: outcome == 'Goal'
      - psxg_faced: suma de PSxG de los shots linked
      - goals_prevented: depende de la columna PSxG usada (ver arriba)
      - errors_leading_to_shot: GK events con outcome de error
      - errors_leading_to_goal: errores donde el shot fue Goal
    """
    if linked.empty:
        return pd.DataFrame()

    def _agg(grp):
        on_target = grp["_shot_outcome"].isin(["Saved", "Goal"]).sum()
        saves = (grp["_shot_outcome"] == "Saved").sum()
        goals = (grp["_shot_outcome"] == "Goal").sum()
        psxg = grp["_psxg"].sum() if "_psxg" in grp.columns else 0.0

        # Errores: GK events con outcome negativo
        err_mask = grp["goalkeeper_outcome"].isin(ERROR_OUTCOMES)
        err_to_shot = int(err_mask.sum())
        err_to_goal = int((err_mask & (grp["_shot_outcome"] == "Goal")).sum())

        # CÁLCULO DE GOALS PREVENTED según columna PSxG
        # shot_gk_shot_stopping_xg_suppression YA es goals prevented
        # Las otras columnas requieren restar goals
        if psxg_col == "shot_gk_shot_stopping_xg_suppression":
            goals_prevented = float(psxg)  # Ya es goals prevented
        else:
            goals_prevented = float(psxg) - int(goals)  # Calcularlo

        return pd.Series({
            "gk_shots_on_target_against": int(on_target),
            "gk_saves": int(saves),
            "gk_save_pct": saves / on_target if on_target > 0 else np.nan,
            "gk_goals_conceded_shots": int(goals),
            "gk_psxg_faced": float(psxg),
            "gk_goals_prevented": goals_prevented,
            "gk_errors_leading_to_shot": err_to_shot,
            "gk_errors_leading_to_goal": err_to_goal,
        })

    result = linked.groupby(pid_col).apply(_agg, include_groups=False).reset_index()
    return result


def _agg_foot_play(df, gk_ids, pid_col):
    """
    Agrega métricas de juego con los pies para arqueros.
    
    Filtro: eventos type == 'Pass' de jugadores GK.
    
    StatsBomb: pass_outcome es NaN para pases completados,
               'Incomplete' para pases incompletos.
    """
    gk_passes = df[
        (df[pid_col].isin(gk_ids)) & (df["type"] == "Pass")
    ].copy()

    if gk_passes.empty:
        return pd.DataFrame()

    # --- Flags por fila ---
    # Completado: pass_outcome es NaN
    gk_passes["_completed"] = gk_passes["pass_outcome"].isna()

    # Long ball: pass_length > umbral
    if "pass_length" in gk_passes.columns:
        gk_passes["_long_ball"] = (
            pd.to_numeric(gk_passes["pass_length"], errors="coerce") > LONG_BALL_M
        )
    else:
        gk_passes["_long_ball"] = False

    # Tercio defensivo: x < umbral
    if "x" in gk_passes.columns:
        gk_passes["_def_third"] = (
            pd.to_numeric(gk_passes["x"], errors="coerce") < DEF_THIRD_X
        )
    else:
        gk_passes["_def_third"] = False

    # Bajo presión
    if "under_pressure" in gk_passes.columns:
        gk_passes["_under_press"] = (
            gk_passes["under_pressure"]
            .astype(str).str.strip().str.lower()
            .isin(["true", "1", "1.0"])
        )
    else:
        gk_passes["_under_press"] = False

    # Combo: bajo presión EN tercio defensivo
    gk_passes["_press_def"] = gk_passes["_def_third"] & gk_passes["_under_press"]

    # OBV
    has_obv = "obv_total_net" in gk_passes.columns
    if has_obv:
        gk_passes["obv_total_net"] = pd.to_numeric(
            gk_passes["obv_total_net"], errors="coerce"
        ).fillna(0)

    def _agg(grp):
        total = len(grp)
        completed = int(grp["_completed"].sum())
        long_balls = int(grp["_long_ball"].sum())
        press_def = int(grp["_press_def"].sum())
        press_def_comp = int((grp["_press_def"] & grp["_completed"]).sum())
        obv = float(grp["obv_total_net"].sum()) if has_obv else 0.0

        return pd.Series({
            "gk_long_ball_pct": long_balls / total if total > 0 else np.nan,
            "gk_pressured_passes_def_third": press_def,
            "gk_pressured_passes_def_third_completion_pct": (
                press_def_comp / press_def if press_def > 0 else np.nan
            ),
        })

    result = gk_passes.groupby(pid_col).apply(_agg, include_groups=False).reset_index()
    return result


def _agg_outside_box(df, gk_ids, pid_col):
    """
    Agrega métricas de juego fuera del área.
    
    - gk_actions_outside_box: eventos del GK donde x > línea del área (18m)
    - gk_aggressive_distance_avg: promedio de x de TODOS los eventos del GK
      (indica qué tan adelantado juega en promedio)
    """
    gk_all = df[df[pid_col].isin(gk_ids)].copy()

    if gk_all.empty or "x" not in gk_all.columns:
        return pd.DataFrame()

    gk_all["_x"] = pd.to_numeric(gk_all["x"], errors="coerce")
    gk_all["_outside"] = gk_all["_x"] > BOX_LINE_X

    def _agg(grp):
        return pd.Series({
            "gk_actions_outside_box": int(grp["_outside"].sum()),
            "gk_aggressive_distance_avg": float(grp["_x"].mean()),
        })

    result = gk_all.groupby(pid_col).apply(_agg, include_groups=False).reset_index()
    return result


def _agg_own_goals(df, gk_ids, pid_col):
    """
    Cuenta autogoles concedidos por cada arquero.
    
    Lógica: para cada partido donde jugó el GK, contar eventos
    'Own Goal Against' donde team == equipo del GK.
    
    En StatsBomb, 'Own Goal Against' tiene team = el equipo que CONCEDIÓ
    el autogol (el equipo contra el cual se anotó).
    """
    # Obtener equipo y partidos de cada GK
    gk_events = df[df[pid_col].isin(gk_ids)]
    if gk_events.empty or "team" not in gk_events.columns:
        return pd.DataFrame(columns=[pid_col, "gk_own_goals_against"])

    # Mapeo: (GK, match) → team
    gk_match_team = (
        gk_events.groupby([pid_col, "match_id"])["team"]
        .first()
        .reset_index()
    )

    # Eventos Own Goal Against
    df_oga = df[df["type"] == "Own Goal Against"]
    if df_oga.empty:
        return pd.DataFrame(columns=[pid_col, "gk_own_goals_against"])

    # Cada OGA tiene (match_id, team). Merge con GK match/team para atribuir.
    oga_lk = df_oga[["match_id", "team"]].copy()
    merged = gk_match_team.merge(oga_lk, on=["match_id", "team"], how="inner")

    if merged.empty:
        return pd.DataFrame(columns=[pid_col, "gk_own_goals_against"])

    result = merged.groupby(pid_col).size().reset_index(name="gk_own_goals_against")
    return result


def _agg_penalties(df, pid_col, psxg_col):
    """
    Calcula métricas específicas de penalties.
    
    Similar a _build_gk_shot_links pero SOLO para penalties.
    Permite analizar el rendimiento del arquero en penalties por separado.
    
    Retorna DataFrame con:
      - gk_penalties_faced: Total de penalties enfrentados
      - gk_penalties_saved: Penalties atajados
      - gk_penalties_conceded: Goles de penalty concedidos
      - gk_penalty_save_pct: % de penalties atajados
      - gk_penalties_psxg: PSxG total de penalties (informativo)
    """
    # Filtrar GK events
    df_gk = df[df["type"] == "Goal Keeper"].copy()
    if df_gk.empty:
        return pd.DataFrame()

    # Parse y explotar related_events
    df_gk["_related"] = df_gk["related_events"].apply(_parse_related)
    df_gk_exp = df_gk.explode("_related").rename(columns={"_related": "_rel_id"})
    df_gk_exp = df_gk_exp[
        df_gk_exp["_rel_id"].notna() & (df_gk_exp["_rel_id"].astype(str).str.len() > 0)
    ]

    if df_gk_exp.empty:
        return pd.DataFrame()

    # Preparar SOLO penalties
    df_penalties = df[(df["type"] == "Shot") & (df["shot_type"] == "Penalty")].copy()
    if df_penalties.empty:
        return pd.DataFrame()

    pen_cols = ["id", "shot_outcome"]
    if psxg_col and psxg_col in df_penalties.columns:
        df_penalties[psxg_col] = pd.to_numeric(df_penalties[psxg_col], errors="coerce").fillna(0)
        pen_cols.append(psxg_col)

    penalties_lk = df_penalties[pen_cols].copy()
    penalties_lk["id"] = penalties_lk["id"].astype(str)

    # Rename para merge
    rename_map = {"id": "_rel_id", "shot_outcome": "_pen_outcome"}
    if psxg_col and psxg_col in penalties_lk.columns:
        rename_map[psxg_col] = "_pen_psxg"
    penalties_lk = penalties_lk.rename(columns=rename_map)

    # Merge: solo penalties vinculados a GK events
    linked = df_gk_exp.merge(penalties_lk, on="_rel_id", how="inner")

    if linked.empty:
        return pd.DataFrame()

    # Agregar por arquero
    def _agg(grp):
        faced = len(grp)
        saved = (grp["_pen_outcome"] == "Saved").sum()
        conceded = (grp["_pen_outcome"] == "Goal").sum()
        psxg = grp["_pen_psxg"].sum() if "_pen_psxg" in grp.columns else 0.0

        return pd.Series({
            "gk_penalties_faced": int(faced),
            "gk_penalties_saved": int(saved),
            "gk_penalties_conceded": int(conceded),
            "gk_penalty_save_pct": saved / faced if faced > 0 else np.nan,
            "gk_penalties_psxg": float(psxg),
        })

    result = linked.groupby(pid_col).apply(_agg, include_groups=False).reset_index()
    return result


# ============= FUNCIÓN PRINCIPAL =============

def calculate_gk_metrics(df, player_minutes_summary):
    """
    Calcula métricas avanzadas para arqueros.
    
    Interface diseñada para ser llamada desde main_analysis.py,
    exactamente como calculate_cb_zone_metrics o calculate_lane_bias_metrics.
    
    Args:
        df: DataFrame de eventos (df_all de main_analysis).
            Columnas necesarias: _player_id/pid, type, id, related_events,
            goalkeeper_outcome, shot_outcome, shot_shot_execution_xg,
            pass_outcome, pass_length, obv_total_net, under_pressure,
            x, match_id, team, position.
        player_minutes_summary: DataFrame de main_analysis.
            Columnas necesarias: player_id, primary_position, total_minutes.

    Returns:
        DataFrame con columnas:
            player_id  +  todas las métricas prefijadas con 'gk_'
        Retorna DataFrame vacío si no hay datos de arqueros.
    """
    print("\n--- Calculando métricas de arqueros ---")

    # --- Validaciones de entrada ---
    pid_col = _get_pid_col(df)
    if pid_col is None:
        print("  ⚠️  No se encontró columna de player_id en df")
        return pd.DataFrame()

    if "id" not in df.columns:
        print("  ⚠️  Columna 'id' (UUID evento) no encontrada. "
              "Se necesita para linking via related_events.")
        return pd.DataFrame()

    # --- Identificar arqueros ---
    gk_ids = _identify_gks(df, player_minutes_summary, pid_col)
    if not gk_ids:
        print("  ⚠️  No se identificaron arqueros")
        return pd.DataFrame()
    print(f"  Arqueros identificados: {len(gk_ids)}")

    # --- Detectar columna PSxG ---
    psxg_col = _get_psxg_col(df[df["type"] == "Shot"])
    if psxg_col:
        print(f"  Columna PSxG: {psxg_col}")
        if psxg_col == "shot_gk_shot_stopping_xg_suppression":
            print(f"    ✓ Usando métrica específica de StatsBomb para shot stopping")
        elif psxg_col == "shot_shot_execution_xg":
            print(f"    ⚠️  Fallback: post-shot xG general")
        elif psxg_col == "shot_statsbomb_xg":
            print(f"    ⚠️  Fallback: pre-shot xG (menos preciso)")
    else:
        print("  ⚠️  No se encontró columna de PSxG. "
              "goals_prevented será = −goals_conceded")

    # --- STEP 1: Linking GK → Shots (núcleo de la precisión) ---
    linked = _build_gk_shot_links(df, pid_col, psxg_col)
    if linked.empty:
        print("  ⚠️  No se generaron links GK→Shot (related_events vacíos o sin shots)")
    else:
        print(f"  Links GK→Shot generados: {len(linked):,}")
        # Informar sobre penalties excluidos
        penalties_total = ((df["type"] == "Shot") & (df["shot_type"] == "Penalty")).sum()
        if penalties_total > 0:
            print(f"  Penalties excluidos completamente: {penalties_total}")

    # --- STEP 2: Agregaciones por categoría ---
    shot_stopping = _agg_shot_stopping(linked, pid_col, psxg_col)
    foot_play    = _agg_foot_play(df, gk_ids, pid_col)
    outside_box  = _agg_outside_box(df, gk_ids, pid_col)
    own_goals    = _agg_own_goals(df, gk_ids, pid_col)
    penalties    = _agg_penalties(df, pid_col, psxg_col)

    # --- STEP 3: Combinar en un solo DataFrame ---
    # Base: todos los GKs que tienen minutos en player_minutes_summary
    minutes_map = player_minutes_summary.set_index("player_id")["total_minutes"].to_dict()

    base_ids = [gid for gid in gk_ids if minutes_map.get(gid, 0) > 0]
    if not base_ids:
        print("  ⚠️  Ningún arquero tiene minutos registrados")
        return pd.DataFrame()

    result = pd.DataFrame({pid_col: base_ids})
    result["_total_minutes"] = result[pid_col].map(minutes_map)
    result["_per90"] = 90.0 / result["_total_minutes"]

    # Merge cada agregación (left join: GKs sin datos en esa categoría quedan NaN)
    agg_groups = [
        (shot_stopping, "shot_stopping"),
        (foot_play,     "foot_play"),
        (outside_box,   "outside_box"),
        (own_goals,     "own_goals"),
        (penalties,     "penalties"),
    ]

    for agg_df, label in agg_groups:
        if not agg_df.empty and pid_col in agg_df.columns:
            result = result.merge(agg_df, on=pid_col, how="left")
            print(f"  ✓ {label}: {agg_df[pid_col].nunique()} arqueros")
        else:
            print(f"  ⚠️  {label}: sin datos")

    # --- STEP 4: Derivar métricas compuestas ---
    # Ensure columns exist (con default 0 si no se generaron)
    for col in ("gk_goals_conceded_shots", "gk_own_goals_against",
                "gk_psxg_faced", "gk_goals_prevented"):
        if col not in result.columns:
            result[col] = 0.0

    result["gk_goals_conceded_shots"] = result["gk_goals_conceded_shots"].fillna(0)
    result["gk_own_goals_against"]    = result["gk_own_goals_against"].fillna(0)
    
    # Goals concedidos total: SOLO goals de shots (ya excluye penalties)
    # NO incluir own goals (no son "culpa" del arquero para métricas de shot stopping)
    result["gk_goals_conceded_total"] = result["gk_goals_conceded_shots"]

    # --- STEP 5: Normalizar per90 ---
    per90_map = {
        "gk_shots_on_target_against_per90": "gk_shots_on_target_against",
        "gk_saves_per90":                   "gk_saves",
        "gk_psxg_faced_per90":              "gk_psxg_faced",
        "gk_goals_prevented_per90":         "gk_goals_prevented",
        "gk_goals_conceded_per90":          "gk_goals_conceded_total",
        "gk_pressured_passes_def_third_per90": "gk_pressured_passes_def_third",
        "gk_actions_outside_box_per90":     "gk_actions_outside_box",
        # Penalties
        "gk_penalties_faced_per90":         "gk_penalties_faced",
        "gk_penalties_saved_per90":         "gk_penalties_saved",
        "gk_penalties_conceded_per90":      "gk_penalties_conceded",
        # Own Goals
        "gk_own_goals_against_per90":       "gk_own_goals_against",
    }

    for per90_col, source_col in per90_map.items():
        if source_col in result.columns:
            result[per90_col] = result[source_col] * result["_per90"]

    # --- STEP 6: Redondeo y limpieza ---
    gk_cols = [c for c in result.columns if c.startswith("gk_")]
    for col in gk_cols:
        if result[col].dtype in (np.float64, np.float32):
            result[col] = result[col].round(4)

    # Rename pid_col → player_id para el merge en main_analysis
    if pid_col != "player_id":
        result = result.rename(columns={pid_col: "player_id"})

    # Seleccionar columnas finales: player_id + métricas de salida
    # (descartar columnas internas: _total_minutes, _per90, y absolutos intermedios)
    output_metrics = [
        # Shot Stopping - Ratios
        "gk_save_pct",
        # Shot Stopping - Totales
        "gk_shots_on_target_against",
        "gk_saves",
        "gk_psxg_faced",
        "gk_goals_prevented",
        "gk_errors_leading_to_shot",
        "gk_errors_leading_to_goal",
        # Shot Stopping - Per90
        "gk_shots_on_target_against_per90",
        "gk_saves_per90",
        "gk_psxg_faced_per90",
        "gk_goals_prevented_per90",
        # Goals Conceded - Totales
        "gk_goals_conceded_shots",
        "gk_goals_conceded_total",
        "gk_own_goals_against",
        # Goals Conceded - Per90
        "gk_goals_conceded_per90",
        "gk_own_goals_against_per90",
        # Penalties - Ratios
        "gk_penalty_save_pct",
        # Penalties - Totales
        "gk_penalties_faced",
        "gk_penalties_saved",
        "gk_penalties_conceded",
        "gk_penalties_psxg",
        # Penalties - Per90
        "gk_penalties_faced_per90",
        "gk_penalties_saved_per90",
        "gk_penalties_conceded_per90",
        # Foot Play - Ratios
        "gk_long_ball_pct",
        "gk_pressured_passes_def_third_completion_pct",
        # Foot Play - Totales
        "gk_pressured_passes_def_third",
        "gk_pass_obv_per90",
        "gk_pressured_passes_def_third_per90",
        # Outside Box - Totales
        "gk_actions_outside_box",
        # Outside Box - Per90 y promedios
        "gk_actions_outside_box_per90",
        "gk_aggressive_distance_avg",
    ]

    # Filtrar solo las que existen en result
    final_cols = ["player_id"] + [c for c in output_metrics if c in result.columns]
    result = result[final_cols].copy()

    # Convertir player_id a Int64 para consistencia con main_analysis
    result["player_id"] = result["player_id"].astype("Int64")

    print(f"  ✓ Arqueros con métricas finales: {len(result):,}")
    print(f"  Columnas exportadas: {list(result.columns[1:])}")

    return result