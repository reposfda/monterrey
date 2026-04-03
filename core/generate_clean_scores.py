"""
Script para generar archivos de scores limpios por posición.

Reutiliza funciones existentes de:
- utils/scoring_wrappers.py (PROFILE_LABELS, _generate_profile_flags)
- config.py (paths, defaults)
- scoring/* (funciones de scoring)

USO:
    python generate_clean_scores.py

Genera archivos limpios (17 columnas) en data/scores/
"""
from pathlib import Path
import pandas as pd

# Importar configuración existente
from config import DATA_DIR, Defaults

# Importar funciones existentes de scoring_wrappers
from utils.scoring_wrappers import PROFILE_LABELS, _generate_profile_flags

# Importar funciones de scoring por posición
from scoring.forwards import run_delantero_scoring, run_extremo_scoring
from scoring.midfielders import run_volante_scoring, run_interior_scoring
from scoring.defenders import run_cb_scoring, score_lateral_df
from scoring.goalkeeper import run_goalkeeper_scoring


# =============================================================================
# CONFIGURACIÓN (usando config.py)
# =============================================================================

# Paths desde config.py
PER90_CSV = DATA_DIR / "per90" / "all_players_complete_2025_2026.csv"
OUTPUT_DIR = DATA_DIR / "scores"

# Filtros desde config.py
MIN_MINUTES = Defaults.MIN_MINUTES  # 450
MIN_MATCHES = Defaults.MIN_MATCHES  # 3

# Columnas base a mantener
BASE_COLS = [
    "player_id",
    "player_name", 
    "team_name",
    "matches",
    "minutes",
    "primary_position",
    "primary_position_share",
]


# =============================================================================
# MAPEO DE POSICIONES
# =============================================================================

POSITION_SCORERS = {
    "Delantero": {
        "function": run_delantero_scoring,
        "output_name": "delantero",
    },
    "Extremo": {
        "function": run_extremo_scoring,
        "output_name": "extremo",
    },
    "Volante": {
        "function": run_volante_scoring,
        "output_name": "volante",
    },
    "Interior/Mediapunta": {
        "function": run_interior_scoring,
        "output_name": "interior",
    },
    "Zaguero": {
        "function": run_cb_scoring,
        "output_name": "zaguero",
    },
    "Lateral": {
        "function": score_lateral_df,
        "output_name": "lateral",
    },
    "Golero": {
        "function": run_goalkeeper_scoring,
        "output_name": "golero",
    },
}


# =============================================================================
# FUNCIÓN PARA SELECCIONAR COLUMNAS FINALES
# =============================================================================

def select_final_columns(df: pd.DataFrame, position_key: str) -> pd.DataFrame:
    """
    Selecciona SOLO las columnas necesarias para el archivo final.
    
    Esta es la ÚNICA función nueva - todo lo demás se reutiliza.
    
    Mantiene:
    - Columnas base (player_id, player_name, etc.)
    - Scores por categoría (Score_*)
    - Flags individuales (renombrados de flag_Score_* a Flag_*)
    - Columna Flags (ya generada por _generate_profile_flags)
    """
    df = df.copy()
    
    # 1. Columnas base
    selected_cols = [col for col in BASE_COLS if col in df.columns]
    
    # 2. Columnas Score_* (EXCEPTO Score_Total, usar Score_Overall)
    score_cols = [
        col for col in df.columns 
        if col.startswith('Score_') and col != 'Score_Total'
    ]
    selected_cols.extend(score_cols)
    
    # 3. Columnas flag_Score_* (las renombraremos)
    flag_cols = [col for col in df.columns if col.startswith('flag_Score_')]
    selected_cols.extend(flag_cols)
    
    # 4. Columna Flags (si existe)
    if 'Flags' in df.columns:
        selected_cols.append('Flags')
    
    # Filtrar DataFrame
    df_clean = df[selected_cols].copy()
    
    # 5. Renombrar flag_Score_* a Flag_* y convertir a booleanos
    rename_map = {}
    for col in df_clean.columns:
        if col.startswith('flag_Score_'):
            new_name = col.replace('flag_Score_', 'Flag_')
            rename_map[col] = new_name
    
    if rename_map:
        df_clean = df_clean.rename(columns=rename_map)
        
        # Convertir a booleanos (True/False)
        for new_col in rename_map.values():
            df_clean[new_col] = df_clean[new_col].astype(int).astype(bool)
    
    # 6. Renombrar Score_Total a Score_Overall si existe
    if 'Score_Total' in df_clean.columns:
        df_clean = df_clean.rename(columns={'Score_Total': 'Score_Overall'})
    
    return df_clean


# =============================================================================
# FUNCIÓN GENÉRICA PARA CUALQUIER POSICIÓN
# =============================================================================

def generate_position_scores(
    position_key: str,
    per90_csv: Path,
    output_file: Path,
    verbose: bool = True
):
    """
    Genera archivo limpio de scores para cualquier posición.
    
    Reutiliza:
    - Funciones de scoring del módulo scoring/
    - _generate_profile_flags() de utils/scoring_wrappers.py
    """
    if position_key not in POSITION_SCORERS:
        raise ValueError(
            f"Posición no reconocida: {position_key}. "
            f"Válidas: {list(POSITION_SCORERS.keys())}"
        )
    
    scorer_info = POSITION_SCORERS[position_key]
    scoring_function = scorer_info["function"]
    
    if verbose:
        print(f"\n{'='*70}")
        print(f"GENERANDO: {output_file.name}")
        print(f"Posición: {position_key}")
        print(f"{'='*70}")
    
    # 1. Calcular scores (reutilizando función existente)
    try:
        # Intentar con per90_csv
        df_full = scoring_function(
            per90_csv=per90_csv,
            min_minutes=MIN_MINUTES,
            min_matches=MIN_MATCHES,
        )
    except TypeError:
        try:
            # Intentar con df
            df_full = scoring_function(
                df=pd.read_csv(per90_csv),
                position_group=position_key,
                min_minutes=MIN_MINUTES,
                min_matches=MIN_MATCHES,
            )
        except TypeError:
            # Para Lateral que usa per90_df
            df_full = scoring_function(
                per90_df=pd.read_csv(per90_csv),
                position_group=position_key,
                min_minutes=MIN_MINUTES,
                min_matches=MIN_MATCHES,
                verbose=False,
            )
    
    # 2. Generar columna Flags (reutilizando función existente)
    df_full = _generate_profile_flags(df_full, position_key)
    
    # 3. Seleccionar columnas finales (ÚNICA función nueva)
    df_clean = select_final_columns(df_full, position_key)
    
    # 4. Guardar
    output_file.parent.mkdir(parents=True, exist_ok=True)
    df_clean.to_csv(output_file, index=False)
    
    if verbose:
        print(f"\n✅ Archivo generado: {output_file}")
        print(f"   Total jugadores: {len(df_clean):,}")
        print(f"   Total columnas: {len(df_clean.columns)}")
        print(f"   Columnas: {list(df_clean.columns)}")


def generate_all_positions(
    per90_csv: Path,
    output_dir: Path,
    season_suffix: str = "2025_2026",
    positions: list[str] | None = None,
):
    """
    Genera archivos limpios para todas las posiciones automáticamente.
    """
    if positions is None:
        positions = list(POSITION_SCORERS.keys())
    
    print(f"\n{'='*70}")
    print(f"GENERANDO SCORES LIMPIOS PARA {len(positions)} POSICIONES")
    print(f"{'='*70}")
    
    for position_key in positions:
        output_name = POSITION_SCORERS[position_key]["output_name"]
        output_file = output_dir / f"{output_name}_scores_{season_suffix}.csv"
        
        try:
            generate_position_scores(
                position_key=position_key,
                per90_csv=per90_csv,
                output_file=output_file,
                verbose=True,
            )
        except Exception as e:
            print(f"\n❌ Error procesando {position_key}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    print(f"\n{'='*70}")
    print(f"✅ PROCESO COMPLETADO")
    print(f"{'='*70}")


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    # Verificar que existe el archivo de entrada
    if not PER90_CSV.exists():
        print(f"❌ Error: No se encontró el archivo {PER90_CSV}")
        print(f"\nPor favor, ejecutá primero:")
        print(f"  python calculate_main_csv.py")
        exit(1)
    
    # Generar TODAS las posiciones automáticamente
    generate_all_positions(
        per90_csv=PER90_CSV,
        output_dir=OUTPUT_DIR,
        season_suffix="2025_2026",
    )
    
    print(f"\nLos archivos generados están listos para usarse con:")
    print(f"  python core/consolidar_scores.py")