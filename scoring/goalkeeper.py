# scoring/goalkeeper.py
# -*- coding: utf-8 -*-
"""
Scoring for Goalkeeper position.

Includes:
- GoalkeeperScorer: Goalkeeper specific metrics
"""
from __future__ import annotations
from pathlib import Path
import pandas as pd

from .base import PositionScorer


# =============================================================================
# GOALKEEPER (Arquero)
# =============================================================================

class GoalkeeperScorer(PositionScorer):
    """
    Scorer for Goleros (goalkeepers).
    
    Categories:
    - Effectiveness (50%): Shot-stopping and error prevention
    - Area Domination (20%): Command of penalty area
    - Foot Play (15%): Distribution and passing
    - Outside Box (15%): Sweeper keeper actions
    """
    
    @property
    def position_group(self) -> str:
        return "Golero"
    
    @property
    def categories(self) -> dict:
        return {
            "Score_Effectiveness": [
                ("gk_goals_prevented_per90", 0.50, False),
                ("gk_save_pct", 0.25, False),
                ("gk_errors_leading_to_shot_per90", 0.10, True),  # inverted
                ("gk_errors_leading_to_goal_per90", 0.15, True),  # inverted
            ],
            "Score_Area_Domination": [
                ("gk_claims_per90", 0.50, False),
                ("gk_shots_open_play_in_box_against_per90", 0.50, True),  # inverted
            ],
            "Score_Foot_Play": [
                ("gk_pass_obv_per90", 0.40, False),
                ("gk_long_ball_pct", 0.20, False),
                ("gk_pressured_passes_def_third_per90", 0.20, False),
                ("gk_pressured_passes_def_third_completion_pct", 0.20, False),
            ],
            "Score_Outside_Box": [
                ("gk_actions_outside_box_per90", 0.50, False),
                ("gk_aggressive_distance_avg", 0.50, False),
            ],
        }
    
    @property
    def category_weights(self) -> dict:
        return {
            "Score_Effectiveness": 0.50,
            "Score_Area_Domination": 0.20,
            "Score_Foot_Play": 0.15,
            "Score_Outside_Box": 0.15,
        }


# =============================================================================
# LEGACY COMPATIBILITY FUNCTION
# =============================================================================

def run_goalkeeper_scoring(
    per90_csv: Path | None = None,
    out_csv: Path | None = None,
    df: pd.DataFrame | None = None,
    position_group: str = "Golero",
    min_minutes: int = 450,
    min_matches: int = 3,
    flag_q: float = 0.75,
) -> pd.DataFrame:
    """
    Legacy wrapper for Goalkeeper scoring (backward compatibility).
    
    Parameters
    ----------
    per90_csv : Path, optional
        Path to per90 CSV file
    out_csv : Path, optional
        Path to save output CSV
    df : pd.DataFrame, optional
        DataFrame with per90 data
    position_group : str, default='Golero'
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
        Scored goalkeepers
    """
    scorer = GoalkeeperScorer(
        min_minutes=min_minutes,
        min_matches=min_matches,
        flag_q=flag_q,
        verbose=True
    )
    return scorer.score(per90_csv=per90_csv, df=df, out_csv=out_csv)
