# -*- coding: utf-8 -*-
"""
DIAGNÓSTICO: Minutos y Partidos Faltantes
==========================================

Este script analiza el dataset de eventos original para identificar por qué
tu script calcula menos minutos y partidos que StatsBomb oficial.

PROBLEMAS DETECTADOS:
- Script: 636,631 min vs StatsBomb: 753,176 min (-15.5%)
- Script: 7,454 apariciones vs StatsBomb: 10,221 apariciones (-27.1%)

POSIBLES CAUSAS A INVESTIGAR:
1. Partidos faltantes en el dataset de eventos
2. Cálculo incorrecto de minutos por substituciones
3. Filtros que excluyen partidos válidos
4. Diferencia en definición de "aparición"

"""

import pandas as pd
import numpy as np
import json, ast
from datetime import datetime
import os

# ============= CONFIGURACIÓN =============
PATH_EVENTS = "../data/events_2024_2025.csv"  # Ajustar a tu ruta
PATH_STATSBOMB = "../data/reaccesostatsbomb/jugador_ligamx_24_25.xlsx"  # Datos oficiales para comparar

# ============= HELPERS (copiados del script original) =============
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

def get_player_id_series(df):
    for cand in ["player_id", "playerId", "idPlayer"]:
        if cand in df.columns:
            return pd.to_numeric(df[cand], errors="coerce")
    if "player" in df.columns:
        obj = df["player"].apply(_coerce_literal)
        return obj.apply(lambda v: v.get("id") if isinstance(v, dict) else np.nan)
    return pd.Series(np.nan, index=df.index)

def get_team_name(val):
    if pd.isna(val):
        return None
    v = _coerce_literal(val)
    if isinstance(v, dict):
        return v.get("name") or v.get("team_name")
    return str(val)

def lineup_players_from_match(df_match):
    """Extrae XI titular del partido"""
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
                if pid is not None:
                    out[int(pid)] = {
                        'team': team_val
                    }
    return out

def match_end_minute(df_match):
    """Final del partido por minuto máximo observado"""
    ASSUME_END_CAP = 120
    mx = pd.to_numeric(df_match.get("minute", pd.Series([], dtype=float)), errors="coerce").max()
    if pd.isna(mx):
        return 90.0
    return float(min(ASSUME_END_CAP, max(90.0, mx)))

print("=" * 80)
print("DIAGNÓSTICO: MINUTOS Y PARTIDOS FALTANTES")
print("=" * 80)

# ============= 1. CARGAR DATOS =============
print("1. Cargando datos...")

try:
    df_events = pd.read_csv(PATH_EVENTS, low_memory=False)
    df_events["_player_id"] = get_player_id_series(df_events)
    df_events["minute"] = pd.to_numeric(df_events.get("minute", np.nan), errors="coerce")
    print(f"✓ Eventos cargados: {len(df_events):,}")
    print(f"  Eventos con player_id: {df_events['_player_id'].notna().sum():,}")
    print(f"  Partidos únicos: {df_events['match_id'].nunique():,}")
except Exception as e:
    print(f"❌ Error cargando eventos: {e}")
    print("   Asegúrate de ajustar PATH_EVENTS")
    exit(1)

try:
    df_sb = pd.read_excel(PATH_STATSBOMB)
    print(f"✓ StatsBomb cargado: {len(df_sb):,} registros")
    print(f"  Jugadores únicos: {df_sb['player_id'].nunique():,}")
    print(f"  Partidos únicos: {df_sb['match_id'].nunique():,}")
except Exception as e:
    print(f"❌ Error cargando StatsBomb: {e}")
    exit(1)

# ============= 2. ANÁLISIS DE PARTIDOS =============
print("\n" + "="*60)
print("2. ANÁLISIS DE PARTIDOS")
print("="*60)

# Partidos en cada dataset
eventos_matches = set(df_events['match_id'].unique())
sb_matches = set(df_sb['match_id'].unique())

print(f"Partidos en eventos: {len(eventos_matches):,}")
print(f"Partidos en StatsBomb: {len(sb_matches):,}")

# Partidos comunes y faltantes
common_matches = eventos_matches & sb_matches
missing_in_eventos = sb_matches - eventos_matches
missing_in_sb = eventos_matches - sb_matches

print(f"Partidos comunes: {len(common_matches):,}")
print(f"Partidos faltantes en eventos: {len(missing_in_eventos):,}")
print(f"Partidos faltantes en StatsBomb: {len(missing_in_sb):,}")

if missing_in_eventos:
    print(f"\n⚠️  PARTIDOS FALTANTES EN TU DATASET (primeros 10):")
    for i, match_id in enumerate(list(missing_in_eventos)[:10]):
        # Buscar info del partido en StatsBomb
        match_info = df_sb[df_sb['match_id'] == match_id]
        if not match_info.empty:
            teams = match_info['team_name'].unique()
            date_info = match_info[['match_id']].iloc[0]
            print(f"  Match {match_id}: {teams}")
        else:
            print(f"  Match {match_id}")

# ============= 3. ANÁLISIS DE MINUTOS POR PARTIDO =============
print("\n" + "="*60)
print("3. ANÁLISIS DE MINUTOS POR PARTIDO")
print("="*60)

# Calcular minutos por partido en eventos (usando lógica del script)
print("Calculando minutos usando lógica del script original...")

match_analysis = []

for match_id in list(common_matches)[:50]:  # Analizar primeros 50 partidos comunes
    df_match = df_events[df_events['match_id'] == match_id].copy()
    
    if df_match.empty:
        continue
    
    # Información básica del partido
    match_end = match_end_minute(df_match)
    xi_players = lineup_players_from_match(df_match)
    
    # Jugadores con eventos en el partido
    players_with_events = df_match['_player_id'].dropna().unique()
    
    # Calcular minutos para cada jugador (simulando script)
    script_minutes_match = 0
    sb_minutes_match = 0
    
    for player_id in players_with_events:
        if pd.isna(player_id):
            continue
        
        player_id_int = int(player_id)
        
        # SCRIPT: Lógica simplificada (asume que si hay eventos = jugó todo)
        # Esto es una simplificación - tu script real es más complejo
        is_starter = player_id_int in xi_players
        if is_starter:
            script_minutes_player = match_end  # Asume jugó completo si es titular
        else:
            # Si no es titular pero tiene eventos, asume que jugó algo
            player_events = df_match[df_match['_player_id'] == player_id_int]
            if not player_events.empty:
                # Estimar minutos basado en primer/último evento
                min_minute = player_events['minute'].min()
                max_minute = player_events['minute'].max()
                if pd.notna(min_minute) and pd.notna(max_minute):
                    script_minutes_player = max_minute - min_minute + 10  # +10 como buffer
                else:
                    script_minutes_player = 30  # Estimación por defecto
            else:
                script_minutes_player = 0
        
        script_minutes_match += script_minutes_player
        
        # STATSBOMB: Minutos reales del partido
        sb_player_match = df_sb[(df_sb['match_id'] == match_id) & (df_sb['player_id'] == player_id_int)]
        if not sb_player_match.empty:
            sb_minutes_player = sb_player_match['player_match_minutes'].iloc[0]
            sb_minutes_match += sb_minutes_player
    
    match_analysis.append({
        'match_id': match_id,
        'script_minutes': script_minutes_match,
        'sb_minutes': sb_minutes_match,
        'diff_minutes': sb_minutes_match - script_minutes_match,
        'diff_pct': ((sb_minutes_match - script_minutes_match) / sb_minutes_match * 100) if sb_minutes_match > 0 else 0,
        'match_end': match_end,
        'players_xi': len(xi_players),
        'players_events': len(players_with_events)
    })

df_match_analysis = pd.DataFrame(match_analysis)

if not df_match_analysis.empty:
    print(f"\\nAnálisis de {len(df_match_analysis)} partidos:")
    print(f"  Minutos script promedio por partido: {df_match_analysis['script_minutes'].mean():.0f}")
    print(f"  Minutos StatsBomb promedio por partido: {df_match_analysis['sb_minutes'].mean():.0f}")
    print(f"  Diferencia promedio: {df_match_analysis['diff_minutes'].mean():.0f} min")
    print(f"  Diferencia promedio %: {df_match_analysis['diff_pct'].mean():.1f}%")
    
    # Top 10 partidos con mayores diferencias
    top_diffs = df_match_analysis.nlargest(10, 'diff_minutes')
    print(f"\\n🔍 TOP 10 PARTIDOS CON MAYORES DIFERENCIAS:")
    for _, row in top_diffs.iterrows():
        print(f"  Match {int(row['match_id'])}: {row['diff_minutes']:.0f} min diff ({row['diff_pct']:.1f}%)")
        print(f"    Script: {row['script_minutes']:.0f} | SB: {row['sb_minutes']:.0f}")

# ============= 4. ANÁLISIS DE JUGADORES ESPECÍFICOS =============
print("\n" + "="*60)
print("4. ANÁLISIS DE JUGADORES CON MAYORES DISCREPANCIAS")
print("="*60)

# Cargar datos agregados del script y StatsBomb
try:
    df_script_final = pd.read_csv("../outputs/all_players_complete_2024_2025.csv")
    print("✓ Datos finales del script cargados")
    
    # Agregar StatsBomb por jugador
    df_sb_agg = df_sb.groupby('player_id').agg({
        'player_match_minutes': 'sum',
        'match_id': 'nunique',
        'player_name': 'first'
    }).reset_index()
    
    # Merge para encontrar discrepancias
    df_player_comp = df_script_final[['player_id', 'player_name', 'total_minutes', 'matches_played']].merge(
        df_sb_agg,
        on='player_id',
        how='inner'
    )
    
    # Calcular diferencias
    df_player_comp['minutes_diff'] = df_player_comp['player_match_minutes'] - df_player_comp['total_minutes']
    df_player_comp['minutes_diff_pct'] = np.where(
        df_player_comp['player_match_minutes'] > 0,
        df_player_comp['minutes_diff'] / df_player_comp['player_match_minutes'] * 100,
        0
    )
    
    df_player_comp['matches_diff'] = df_player_comp['match_id'] - df_player_comp['matches_played']
    
    # Top jugadores con mayores discrepancias en minutos
    top_minutes_diff = df_player_comp.nlargest(10, 'minutes_diff')
    
    print(f"\\n🔍 TOP 10 JUGADORES CON MINUTOS FALTANTES:")
    for _, row in top_minutes_diff.iterrows():
        print(f"  {row['player_name_x'][:25]:25} | Diff: {row['minutes_diff']:4.0f} min ({row['minutes_diff_pct']:5.1f}%)")
        print(f"    Script: {row['total_minutes']:4.0f} | SB: {row['player_match_minutes']:4.0f}")
        
        # Buscar eventos de este jugador
        player_matches = df_events[df_events['_player_id'] == row['player_id']]['match_id'].nunique()
        print(f"    Partidos con eventos: {player_matches} | SB dice: {row['match_id']}")
        print()

except Exception as e:
    print(f"⚠️  No se pudo hacer análisis de jugadores: {e}")

# ============= 5. RECOMENDACIONES ESPECÍFICAS =============
print("\n" + "="*60)
print("5. RECOMENDACIONES ESPECÍFICAS")
print("="*60)

print("🎯 ACCIONES RECOMENDADAS:")

if missing_in_eventos:
    print(f"\\n❌ PROBLEMA #1: PARTIDOS FALTANTES")
    print(f"   Tienes {len(missing_in_eventos)} partidos menos que StatsBomb")
    print(f"   Acción: Verificar que tu dataset incluya todos los partidos de la temporada")

if not df_match_analysis.empty and df_match_analysis['diff_pct'].mean() > 10:
    print(f"\\n❌ PROBLEMA #2: CÁLCULO DE MINUTOS")
    print(f"   Diferencia promedio: {df_match_analysis['diff_pct'].mean():.1f}% menos minutos por partido")
    print(f"   Acción: Revisar lógica de substituciones en tu script")
    print(f"   Especialmente las funciones:")
    print(f"   - minutes_played_in_match()")
    print(f"   - sub_on_minute() / sub_off_minute()")

print(f"\\n💡 PRÓXIMOS PASOS:")
print(f"1. Verificar dataset completo: ¿Tienes los {len(sb_matches)} partidos?")
print(f"2. Revisar cálculo de minutos para jugadores suplentes")
print(f"3. Validar detección de substituciones")
print(f"4. Comprobar casos edge (jugadores que entran y salen)")

print(f"\\n📋 DIAGNÓSTICO COMPLETADO")
print(f"Ejecuta este script con tus datos reales para obtener el diagnóstico específico")
print("=" * 80)