# scoring/midfielders.py
# -*- coding: utf-8 -*-
"""
Scoring for Midfielder positions (Volantes e Interiores/Mediapuntas).

Includes:
- VolanteScorer: Defensive and holding midfielders
- InteriorScorer: Attacking midfielders and #10s
"""
from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd

from .base import PositionScorer


# =============================================================================
# VOLANTE (Defensive Midfielder)
# =============================================================================

class VolanteScorer(PositionScorer):
    """
    Scorer for Volantes (defensive/holding midfielders).
    
    Categories:
    - Posesión (25%): Passing and ball retention
    - Progresión (30%): Progressive passes and carries
    - Territoriales (25%): Pressing and recoveries across thirds
    - Contención (20%): Tackling and interceptions
    """
    
    @property
    def position_group(self) -> str:
        return "Volante"
    
    @property
    def categories(self) -> dict:
        return {
            "Score_Posesion": [
                ("complete_passes_per90", 0.20, False),
                ("completed_passes_under_pressure_per90", 0.30, False),
                ("total_turnovers_per90", 0.20, True),  # inverted
                ("obv_total_net_type_pass_per90", 0.30, False),
            ],
            "Score_Progresion": [
                ("pass_into_final_third_per90", 0.20, False),
                ("carry_into_final_third_per90", 0.20, False),
                ("obv_total_net_type_carry_per90", 0.20, False),
                ("pass_switch_per90", 0.20, False),
                ("pass_through_ball_per90", 0.20, False),
            ],
            "Score_Territoriales": [
                ("n_events_third_defensive_pressure_per90", 0.12, False),
                ("n_events_third_middle_pressure_per90", 0.18, False),
                ("counterpress_per90", 0.05, False),
                ("n_events_third_defensive_ball_recovery_per90", 0.15, False),
                ("n_events_third_middle_ball_recovery_per90", 0.20, False),
                ("obv_total_net_type_interception_per90", 0.15, False),
                ("obv_total_net_type_ball_recovery_per90", 0.15, False),
            ],
            "Score_Contencion": [
                ("duel_tackle_per90", 0.22, False),
                ("obv_total_net_duel_type_tackle_per90", 0.23, False),
                ("interception_success_rate", 0.17, False),
                ("obv_total_net_type_interception_per90", 0.23, False),
                ("n_events_third_defensive_interception_per90", 0.15, False),
            ],
        }
    
    @property
    def category_weights(self) -> dict:
        return {
            "Score_Posesion": 0.25,
            "Score_Progresion": 0.30,
            "Score_Territoriales": 0.25,
            "Score_Contencion": 0.20,
        }
    
    def calculate_derived_metrics(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate derived metrics for Volante with fallbacks."""
        if self.verbose:
            print("\n🔧 Calculando métricas derivadas...")
        
        # Fallback for complete_passes_per90
        if "complete_passes_per90" not in df.columns:
            if self.verbose:
                print("  ⚠️  complete_passes_per90 no encontrado")
            
            candidates = ["passes_completed_per90", "completed_passes"]
            for c in candidates:
                if c in df.columns:
                    df["complete_passes_per90"] = df[c]
                    if self.verbose:
                        print(f"  ✓ complete_passes_per90 mapeado desde '{c}'")
                    break
        
        # Fallback for completed_passes_under_pressure_per90
        if "completed_passes_under_pressure_per90" not in df.columns:
            if self.verbose:
                print("  ⚠️  completed_passes_under_pressure_per90 no encontrado")
            
            candidates = ["passes_under_pressure_completed_per90"]
            for c in candidates:
                if c in df.columns:
                    df["completed_passes_under_pressure_per90"] = df[c]
                    if self.verbose:
                        print(f"  ✓ completed_passes_under_pressure_per90 mapeado desde '{c}'")
                    break
        
        # Fallback for duel_tackle_per90
        if "duel_tackle_per90" not in df.columns:
            if self.verbose:
                print("  ⚠️  duel_tackle_per90 no encontrado")
            
            if "tackles_per90" in df.columns:
                df["duel_tackle_per90"] = df["tackles_per90"]
                if self.verbose:
                    print("  ✓ duel_tackle_per90 mapeado desde 'tackles_per90'")
        
        return df


# =============================================================================
# INTERIOR/MEDIAPUNTA (Attacking Midfielder)
# =============================================================================

class InteriorScorer(PositionScorer):
    """
    Scorer for Interiores/Mediapuntas (attacking midfielders, #10s).
    
    Categories:
    - Box to Box (25%): All-around presence across the pitch
    - Desequilibrio (30%): Dribbling and disruption
    - Organización (25%): Passing and chance creation
    - Contención y Presión (20%): Defensive work and pressing
    """
    
    @property
    def position_group(self) -> str:
        return "Interior/Mediapunta"
    
    @property
    def categories(self) -> dict:
        return {
            "Score_BoxToBox": [
                ("n_events_third_defensive_ball_recovery_per90", 0.11, False),
                ("n_events_third_middle_ball_recovery_per90", 0.14, False),
                ("n_events_third_attacking_ball_recovery_per90", 0.11, False),
                ("n_events_third_defensive_duel_per90", 0.08, False),
                ("n_events_third_middle_duel_per90", 0.14, False),
                ("n_events_third_attacking_duel_per90", 0.08, False),
                ("carry_into_final_third_per90", 0.10, False),
                ("touches_in_opp_box_per90", 0.10, False),
                ("shot_touch_pct", 0.05, False),
                ("total_touches_per90", 0.09, False),
            ],
            "Score_Desequilibrio": [
                ("obv_total_net_type_dribble_per90", 0.30, False),
                ("obv_total_net_type_carry_per90", 0.25, False),
                ("carry_into_final_third_per90", 0.15, False),
                ("pass_into_final_third_per90", 0.10, False),
                ("obv_total_net_type_shot_per90", 0.10, False),
                ("shot_statsbomb_xg_per90", 0.10, False),
            ],
            "Score_Organizacion": [
                ("obv_total_net_type_pass_per90", 0.30, False),
                ("complete_passes_per90", 0.20, False),
                ("pass_shot_assist_per90", 0.12, False),
                ("obv_total_net_third_attacking_pass_per90", 0.13, False),
                ("obv_total_net_play_pattern_regular_play_per90", 0.10, False),
                ("total_turnovers_per90", 0.15, True),  # inverted
            ],
            "Score_ContencionPresion": [
                ("n_events_third_middle_pressure_per90", 0.18, False),
                ("n_events_third_attacking_pressure_per90", 0.12, False),
                ("counterpress_per90", 0.10, False),
                ("n_events_third_middle_ball_recovery_per90", 0.12, False),
                ("n_events_third_attacking_ball_recovery_per90", 0.13, False),
                ("obv_total_net_duel_type_tackle_per90", 0.10, False),
                ("duel_tackle_per90", 0.10, False),
                ("obv_total_net_type_interception_per90", 0.08, False),
                ("obv_total_net_third_middle_interception_per90", 0.07, False),
            ],
        }
    
    @property
    def category_weights(self) -> dict:
        return {
            "Score_BoxToBox": 0.25,
            "Score_Desequilibrio": 0.30,
            "Score_Organizacion": 0.25,
            "Score_ContencionPresion": 0.20,
        }
    
    def calculate_derived_metrics(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate derived metrics for Interior/Mediapunta."""
        if self.verbose:
            print("\n🔧 Calculando métricas derivadas...")
        
        # Fallback for complete_passes_per90
        if "complete_passes_per90" not in df.columns:
            if "complete_passes" in df.columns and "minutes" in df.columns:
                df["complete_passes"] = self.safe_numeric(df["complete_passes"])
                df["complete_passes_per90"] = np.where(
                    df["minutes"] > 0,
                    (df["complete_passes"] / df["minutes"]) * 90,
                    np.nan,
                )
                if self.verbose:
                    print("  ✓ complete_passes_per90 calculado desde complete_passes")
        
        return df


# =============================================================================
# LEGACY COMPATIBILITY FUNCTIONS
# =============================================================================

def run_volante_scoring(
    per90_csv: Path | None = None,
    out_csv: Path | None = None,
    df: pd.DataFrame | None = None,
    position_group: str = "Volante",
    min_minutes: int = 450,
    min_matches: int = 3,
    flag_q: float = 0.75,
) -> pd.DataFrame:
    """Legacy wrapper for Volante scoring (backward compatibility)."""
    scorer = VolanteScorer(
        min_minutes=min_minutes,
        min_matches=min_matches,
        flag_q=flag_q,
        verbose=True
    )
    return scorer.score(per90_csv=per90_csv, df=df, out_csv=out_csv)


def score_volante_df(
    per90_df: pd.DataFrame,
    position_group: str = "Volante",
    min_minutes: int = 450,
    min_matches: int = 3,
    flag_q: float = 0.75,
    verbose: bool = True,
) -> pd.DataFrame:
    """Alternative legacy wrapper accepting DataFrame directly."""
    scorer = VolanteScorer(
        min_minutes=min_minutes,
        min_matches=min_matches,
        flag_q=flag_q,
        verbose=verbose
    )
    return scorer.score(df=per90_df)


def run_interior_scoring(
    per90_csv: Path | None = None,
    out_csv: Path | None = None,
    df: pd.DataFrame | None = None,
    position_group: str = "Interior/Mediapunta",
    min_minutes: int = 450,
    min_matches: int = 3,
    flag_q: float = 0.75,
) -> pd.DataFrame:
    """Legacy wrapper for Interior scoring (backward compatibility)."""
    scorer = InteriorScorer(
        min_minutes=min_minutes,
        min_matches=min_matches,
        flag_q=flag_q,
        verbose=True
    )
    return scorer.score(per90_csv=per90_csv, df=df, out_csv=out_csv)
