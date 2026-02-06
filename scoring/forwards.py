# scoring/forwards.py
# -*- coding: utf-8 -*-
"""
Scoring for Forward positions (Delanteros y Extremos).

Includes:
- DelanteroScorer: Classic #9 striker
- ExtremoScorer: Wingers and wide forwards
"""
from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd

from .base import PositionScorer


# =============================================================================
# DELANTERO (Striker)
# =============================================================================

class DelanteroScorer(PositionScorer):
    """
    Scorer for Delanteros (strikers).
    
    Categories:
    - Finalización (40%): Shooting and finishing
    - Presionante (10%): Pressing and defensive work
    - Conector (25%): Passing and link-up play
    - Disruptivo (25%): Dribbling and progressive actions
    """
    
    @property
    def position_group(self) -> str:
        return "Delantero"
    
    @property
    def categories(self) -> dict:
        return {
            "Score_Finalizacion": [
                ("xg_per_shot", 0.20, False),
                ("shot_statsbomb_xg_per90", 0.18, False),
                ("obv_total_net_type_shot_per90", 0.15, False),
                ("goals_per90", 0.05, False),
                ("touches_in_opp_box_per90", 0.15, False),
                ("touches_in_opp_box_pct", 0.10, False),
                ("obv_total_net_play_pattern_regular_play_per90", 0.10, False),
                ("total_shots_per90", 0.04, False),
                ("shot_touch_pct", 0.03, False),
            ],
            "Score_Presionante": [
                ("pressure_per90", 0.30, False),
                ("n_events_third_attacking_pressure_per90", 0.20, False),
                ("counterpress_per90", 0.15, False),
                ("ball_recovery_offensive_per90", 0.15, False),
                ("n_events_third_attacking_ball_recovery_per90", 0.10, False),
                ("obv_total_net_type_interception_per90", 0.05, False),
                ("obv_total_net_type_ball_recovery_per90", 0.05, False),
            ],
            "Score_Conector": [
                ("complete_passes_per90", 0.25, False),
                ("pass_completion_rate", 0.15, False),
                ("pass_into_final_third_per90", 0.15, False),
                ("obv_total_net_type_pass_per90", 0.25, False),
                ("pass_shot_assist_per90", 0.15, False),
                ("total_turnovers_per90", 0.05, True),  # inverted
            ],
            "Score_Disruptivo": [
                ("obv_total_net_type_dribble_per90", 0.40, False),
                ("obv_total_net_type_carry_per90", 0.35, False),
                ("carry_into_final_third_per90", 0.15, False),
                ("pass_into_final_third_per90", 0.10, False),
            ],
        }
    
    @property
    def category_weights(self) -> dict:
        return {
            "Score_Finalizacion": 0.40,
            "Score_Presionante": 0.10,
            "Score_Conector": 0.25,
            "Score_Disruptivo": 0.25,
        }
    
    def calculate_derived_metrics(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate xg_per_shot if missing."""
        if self.verbose:
            print("\n🔧 Calculando métricas derivadas...")
        
        # xg_per_shot
        if "xg_per_shot" not in df.columns:
            if self.verbose:
                print("  ⚠️  xg_per_shot no encontrado, calculando...")
            
            if "shot_statsbomb_xg_per90" in df.columns and "total_shots_per90" in df.columns:
                df["shot_statsbomb_xg_per90"] = self.safe_numeric(df["shot_statsbomb_xg_per90"])
                df["total_shots_per90"] = self.safe_numeric(df["total_shots_per90"])
                df["xg_per_shot"] = np.where(
                    df["total_shots_per90"] > 0,
                    df["shot_statsbomb_xg_per90"] / df["total_shots_per90"],
                    np.nan,
                )
                if self.verbose:
                    print("  ✓ xg_per_shot calculado")
            else:
                if self.verbose:
                    print("  ⚠️  No se pudo calcular xg_per_shot (faltan columnas)")
        
        return df


# =============================================================================
# EXTREMO (Winger)
# =============================================================================

class ExtremoScorer(PositionScorer):
    """
    Scorer for Extremos (wingers).
    
    Categories:
    - Compromiso Defensivo (20%): Pressing and defensive actions
    - Desequilibrio (35%): Dribbling and creative actions
    - Finalización (30%): Shooting and finishing
    - Zona de Influencia (15%): Positional influence and crosses
    """
    
    @property
    def position_group(self) -> str:
        return "Extremo"
    
    @property
    def categories(self) -> dict:
        return {
            "Score_CompromisoDef": [
                ("n_events_third_attacking_pressure_per90", 0.20, False),
                ("counterpress_per90", 0.20, False),
                ("n_events_third_attacking_ball_recovery_per90", 0.15, False),
                ("obv_total_net_type_ball_recovery_per90", 0.20, False),
                ("obv_total_net_type_interception_per90", 0.10, False),
                ("pressure_per90", 0.15, False),
            ],
            "Score_Desequilibrio": [
                ("obv_total_net_type_dribble_per90", 0.18, False),
                ("obv_total_net_type_carry_per90", 0.18, False),
                ("carry_into_final_third_per90", 0.10, False),
                ("pass_into_final_third_per90", 0.08, False),
                ("pass_shot_assist_per90", 0.15, False),
                ("pass_goal_assist_per90", 0.05, False),
                ("xa_per90", 0.12, False),
                ("obv_total_net_type_pass_per90", 0.14, False),
            ],
            "Score_Finalizacion": [
                ("shot_statsbomb_xg_per90", 0.35, False),
                ("obv_total_net_type_shot_per90", 0.25, False),
                ("xg_per_shot", 0.20, False),
                ("touches_in_opp_box_per90", 0.20, False),
            ],
            "Score_ZonaInfluencia": [
                ("obv_from_ext_per90", 0.35, False),
                ("obv_from_int_per90", 0.35, False),
                ("obv_total_net_type_pass_per90", 0.30, False),
            ],
        }
    
    @property
    def category_weights(self) -> dict:
        return {
            "Score_CompromisoDef": 0.20,
            "Score_Desequilibrio": 0.35,
            "Score_Finalizacion": 0.30,
            "Score_ZonaInfluencia": 0.15,
        }
    
    def calculate_derived_metrics(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate derived metrics for wingers."""
        if self.verbose:
            print("\n🔧 Calculando métricas derivadas...")
        
        # xg_per_shot
        if "xg_per_shot" not in df.columns:
            if self.verbose:
                print("  ⚠️  xg_per_shot no encontrado, calculando...")
            
            if "shot_statsbomb_xg_per90" in df.columns and "total_shots_per90" in df.columns:
                df["shot_statsbomb_xg_per90"] = self.safe_numeric(df["shot_statsbomb_xg_per90"])
                df["total_shots_per90"] = self.safe_numeric(df["total_shots_per90"])
                df["xg_per_shot"] = np.where(
                    df["total_shots_per90"] > 0,
                    df["shot_statsbomb_xg_per90"] / df["total_shots_per90"],
                    np.nan,
                )
                if self.verbose:
                    print("  ✓ xg_per_shot calculado")
        
        # Lane influence profile (descriptive)
        if "lane_bias_index" in df.columns:
            if self.verbose:
                print("\n🎯 Calculando perfil de influencia por carriles...")
            
            df["lane_bias_index"] = self.safe_numeric(df["lane_bias_index"])
            
            df["lane_influence_side"] = np.where(
                df["lane_bias_index"].isna(),
                "Sin dato",
                np.where(df["lane_bias_index"] >= 0, "Interior", "Exterior"),
            )
            
            # Lane profile tag
            def lane_tag(row):
                bias = row.get("lane_bias_index", np.nan)
                if pd.isna(bias):
                    return "Sin dato"
                abs_bias = abs(bias)
                if abs_bias < 0.10:
                    return "Balanceado"
                elif abs_bias < 0.25:
                    side = "Interior" if bias >= 0 else "Exterior"
                    return f"Tendencia {side}"
                else:
                    side = "Interior" if bias >= 0 else "Exterior"
                    return f"Marcado {side}"
            
            df["Lane_Profile"] = df.apply(lane_tag, axis=1)
            
            if self.verbose:
                has_bias = int(df["lane_bias_index"].notna().sum())
                if has_bias > 0:
                    ext_count = int((df["lane_influence_side"] == "Exterior").sum())
                    int_count = int((df["lane_influence_side"] == "Interior").sum())
                    print(f"  ✓ Jugadores con perfil de carril: {has_bias}")
                    print(f"    - Perfil Exterior: {ext_count}")
                    print(f"    - Perfil Interior: {int_count}")
        else:
            if self.verbose:
                print("  ⚠️  lane_bias_index no encontrado (métricas de carriles no disponibles)")
            df["lane_influence_side"] = "Sin dato"
            df["Lane_Profile"] = "Sin dato"
        
        return df


# =============================================================================
# LEGACY COMPATIBILITY FUNCTIONS
# =============================================================================

def run_delantero_scoring(
    per90_csv: Path | None = None,
    out_csv: Path | None = None,
    df: pd.DataFrame | None = None,
    position_group: str = "Delantero",
    min_minutes: int = 450,
    min_matches: int = 3,
    flag_q: float = 0.75,
) -> pd.DataFrame:
    """
    Legacy wrapper for Delantero scoring (backward compatibility).
    
    Parameters
    ----------
    per90_csv : Path, optional
        Path to per90 CSV file
    out_csv : Path, optional
        Path to save output CSV
    df : pd.DataFrame, optional
        DataFrame with per90 data
    position_group : str, default='Delantero'
        Position group name
    min_minutes : int, default=450
        Minimum minutes played
    min_matches : int, default=3
        Minimum matches played
    flag_q : float, default=0.75
        Quantile for flags
    
    Returns
    -------
    pd.DataFrame
        Scored players
    """
    scorer = DelanteroScorer(
        min_minutes=min_minutes,
        min_matches=min_matches,
        flag_q=flag_q,
        verbose=True
    )
    return scorer.score(per90_csv=per90_csv, df=df, out_csv=out_csv)


def run_extremo_scoring(
    per90_csv: Path | None = None,
    out_csv: Path | None = None,
    df: pd.DataFrame | None = None,
    position_group: str = "Extremo",
    min_minutes: int = 450,
    min_matches: int = 3,
    flag_q: float = 0.75,
) -> pd.DataFrame:
    """
    Legacy wrapper for Extremo scoring (backward compatibility).
    
    Parameters
    ----------
    per90_csv : Path, optional
        Path to per90 CSV file
    out_csv : Path, optional
        Path to save output CSV
    df : pd.DataFrame, optional
        DataFrame with per90 data
    position_group : str, default='Extremo'
        Position group name
    min_minutes : int, default=450
        Minimum minutes played
    min_matches : int, default=3
        Minimum matches played
    flag_q : float, default=0.75
        Quantile for flags
    
    Returns
    -------
    pd.DataFrame
        Scored players
    """
    scorer = ExtremoScorer(
        min_minutes=min_minutes,
        min_matches=min_matches,
        flag_q=flag_q,
        verbose=True
    )
    return scorer.score(per90_csv=per90_csv, df=df, out_csv=out_csv)
