# scoring/defenders.py
# -*- coding: utf-8 -*-
"""
Scoring for Defender positions (Defensores Centrales y Laterales).

Includes:
- DefensorCentralScorer: Center backs
- LateralScorer: Fullbacks
"""
from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd

from .base import PositionScorer


# =============================================================================
# DEFENSOR CENTRAL (Center Back)
# =============================================================================

class DefensorCentralScorer(PositionScorer):
    """
    Scorer for Defensores Centrales (center backs).
    
    Categories:
    - Acción Defensiva (25%): Defensive actions execution
    - Control Defensivo (45%): Defensive positioning and prevention
    - Progresión (20%): Progressive passing
    - Impacto Ofensivo (10%): Offensive contribution
    """
    
    @property
    def position_group(self) -> str:
        return "Zaguero"
    
    @property
    def categories(self) -> dict:
        return {
            "Score_AccionDefensiva": [
                ("duel_success_rate", 0.25, False),
                ("tackle_success_pct", 0.25, False),
                ("ball_recovery_success_pct", 0.15, False),
                ("interception_success_rate", 0.10, False),
                ("clearances_total_per90", 0.10, False),
                ("blocks_total_per90", 0.05, False),
                ("defensive_actions_lost_per90", 0.10, True),  # inverted
            ],
            "Score_ControlDefensivo": [
                ("pressure_per90", 0.15, False),
                ("counterpress_per90", 0.15, False),
                ("obv_into_per90", 0.25, True),  # inverted (less is better)
                ("obv_from_per90", 0.25, True),  # inverted
                ("shots_from_area_per90", 0.10, True),  # inverted
                ("xg_from_area_per90", 0.10, True),  # inverted
            ],
            "Score_Progresion": [
                ("pass_completion_rate", 0.15, False),
                ("pass_into_final_third_per90", 0.20, False),
                ("pass_switch_per90", 0.20, False),
                ("carry_into_final_third_per90", 0.10, False),
                ("pass_through_ball_per90", 0.15, False),
                ("obv_total_net_type_pass_per90", 0.20, False),
            ],
            "Score_ImpactoOfensivo": [
                ("shot_statsbomb_xg_play_pattern_regular_play_per90", 0.15, False),
                ("shot_statsbomb_xg_play_pattern_from_corner_per90", 0.075, False),
                ("shot_statsbomb_xg_play_pattern_from_free_kick_per90", 0.075, False),
                ("obv_total_net_play_pattern_regular_play_per90", 0.15, False),
                ("obv_total_net_play_pattern_from_free_kick_per90", 0.075, False),
                ("obv_total_net_play_pattern_from_corner_per90", 0.075, False),
            ],
        }
    
    @property
    def category_weights(self) -> dict:
        return {
            "Score_AccionDefensiva": 0.25,
            "Score_ControlDefensivo": 0.45,
            "Score_Progresion": 0.20,
            "Score_ImpactoOfensivo": 0.10,
        }


# =============================================================================
# LATERAL (Fullback)
# =============================================================================

class LateralScorer(PositionScorer):
    """
    Scorer for Laterales (fullbacks).
    
    Categories:
    - Profundidad (30%): Attacking depth and width
    - Calidad (30%): Quality of actions (OBV)
    - Presión (20%): Pressing and defensive intensity
    - Defensivo (20%): Defensive execution (composed from DEF_EXEC + DEF_OBV)
    
    Note: Score_Defensivo is computed as weighted combination of
    defensive execution and defensive OBV subcategories.
    """
    
    def __init__(
        self,
        min_minutes: int = 450,
        min_matches: int = 3,
        flag_q: float = 0.75,
        verbose: bool = True,
        def_exec_w: float = 0.60,
        def_obv_w: float = 0.40,
    ):
        """
        Initialize LateralScorer with defensive sub-weights.
        
        Parameters
        ----------
        def_exec_w : float, default=0.60
            Weight for defensive execution in Score_Defensivo
        def_obv_w : float, default=0.40
            Weight for defensive OBV in Score_Defensivo
        """
        super().__init__(min_minutes, min_matches, flag_q, verbose)
        self.def_exec_w = def_exec_w
        self.def_obv_w = def_obv_w
    
    @property
    def position_group(self) -> str:
        return "Lateral"
    
    @property
    def categories(self) -> dict:
        return {
            "Score_Profundidad": [
                ("pass_into_final_third_per90", 0.15, False),
                ("carry_into_final_third_per90", 0.25, False),
                ("n_events_third_attacking_pass_per90", 0.20, False),
                ("n_events_third_attacking_pass_cross_openplay_per90", 0.20, False),
                ("xa_per90", 0.20, False),
            ],
            "Score_Calidad": [
                ("obv_total_net_type_pass_per90", 0.45, False),
                ("obv_total_net_type_dribble_per90", 0.25, False),
                ("obv_total_net_type_carry_per90", 0.20, False),
                ("total_turnovers_per90", 0.10, True),  # inverted
            ],
            "Score_Presion": [
                ("pressure_per90", 0.35, False),
                ("n_events_third_attacking_pressure_per90", 0.35, False),
                ("ball_recovery_offensive_per90", 0.15, False),
                ("counterpress_per90", 0.15, False),
            ],
            # Defensive subcategories (not in final output, combined into Score_Defensivo)
            "_DEF_EXEC": [
                ("duel_success_rate", 0.25, False),
                ("tackle_success_pct", 0.25, False),
                ("ball_recovery_success_pct", 0.15, False),
                ("interception_success_rate", 0.10, False),
                ("clearances_total_per90", 0.10, False),
                ("blocks_total_per90", 0.05, False),
                ("defensive_actions_lost_per90", 0.10, True),  # inverted
            ],
            "_DEF_OBV": [
                ("obv_total_net_type_duel_per90", 0.20, False),
                ("obv_total_net_duel_type_tackle_per90", 0.25, False),
                ("obv_total_net_type_interception_per90", 0.20, False),
                ("obv_total_net_type_ball_recovery_per90", 0.20, False),
                ("obv_total_net_type_clearance_per90", 0.15, False),
            ],
        }
    
    @property
    def category_weights(self) -> dict:
        return {
            "Score_Profundidad": 0.30,
            "Score_Calidad": 0.30,
            "Score_Presion": 0.20,
            "Score_Defensivo": 0.20,
        }
    
    def score(self, per90_csv: Path | None = None, df: pd.DataFrame | None = None, out_csv: Path | None = None) -> pd.DataFrame:
        """
        Override score() to handle custom defensive category composition.
        """
        # Run base scoring (this calculates _DEF_EXEC and _DEF_OBV)
        df_scored = super().score(per90_csv=per90_csv, df=df, out_csv=None)
        
        # Compose Score_Defensivo from subcategories
        if self.verbose:
            print("\n🔧 Componiendo Score_Defensivo (Exec + OBV)...")
        
        if "_DEF_EXEC" in df_scored.columns and "_DEF_OBV" in df_scored.columns:
            df_scored["Score_Defensivo"] = (
                df_scored["_DEF_EXEC"] * self.def_exec_w +
                df_scored["_DEF_OBV"] * self.def_obv_w
            )
            
            if self.verbose:
                avg = df_scored["Score_Defensivo"].mean()
                print(f"  ✓ Score_Defensivo: promedio = {avg:.1f}")
            
            # Drop internal subcategories
            df_scored = df_scored.drop(columns=["_DEF_EXEC", "_DEF_OBV"], errors="ignore")
        else:
            if self.verbose:
                print("  ⚠️  No se pudieron componer categorías defensivas")
        
        # Recalculate total score with proper weights (now including Score_Defensivo)
        if self.verbose:
            print("\n🎯 Recalculando score total con Score_Defensivo...")
        
        cat_weights = [
            (cat, weight)
            for cat, weight in self.category_weights.items()
            if cat in df_scored.columns
        ]
        
        df_scored["Score_Total"] = self.wavg(df_scored, cat_weights)
        
        if self.verbose:
            avg = df_scored["Score_Total"].mean()
            print(f"  ✓ Score_Total: promedio = {avg:.1f}")
        
        # Re-sort by total score
        df_scored = df_scored.sort_values("Score_Total", ascending=False)
        
        # Regenerate flags with updated Score_Defensivo
        df_scored = self._generate_flags(df_scored)
        
        # Save if requested
        if out_csv:
            df_scored.to_csv(out_csv, index=False, encoding="utf-8")
            if self.verbose:
                print(f"\n💾 Guardado en: {out_csv}")
        
        if self.verbose:
            print(f"\n✅ Scoring completado: {len(df_scored):,} jugadores")
            print("=" * 70)
        
        return df_scored


# =============================================================================
# LEGACY COMPATIBILITY FUNCTIONS
# =============================================================================

def run_cb_scoring(
    per90_csv: Path | None = None,
    out_csv: Path | None = None,
    df: pd.DataFrame | None = None,
    position_group: str = "Zaguero",
    min_minutes: int = 600,
    min_matches: int = 5,
    flag_q: float = 0.75,
) -> pd.DataFrame:
    """Legacy wrapper for Defensor Central scoring (backward compatibility)."""
    scorer = DefensorCentralScorer(
        min_minutes=min_minutes,
        min_matches=min_matches,
        flag_q=flag_q,
        verbose=True
    )
    return scorer.score(per90_csv=per90_csv, df=df, out_csv=out_csv)


def run_position_scoring(
    per90_csv: Path | None = None,
    out_csv: Path | None = None,
    df: pd.DataFrame | None = None,
    position_group: str = "Lateral",
    min_minutes: int = 450,
    min_matches: int = 3,
    flag_q: float = 0.75,
    def_exec_w: float = 0.60,
    def_obv_w: float = 0.40,
) -> pd.DataFrame:
    """Legacy wrapper for Lateral scoring (backward compatibility)."""
    scorer = LateralScorer(
        min_minutes=min_minutes,
        min_matches=min_matches,
        flag_q=flag_q,
        verbose=True,
        def_exec_w=def_exec_w,
        def_obv_w=def_obv_w,
    )
    return scorer.score(per90_csv=per90_csv, df=df, out_csv=out_csv)


def score_lateral_df(
    per90_df: pd.DataFrame,
    position_group: str = "Lateral",
    min_minutes: int = 450,
    min_matches: int = 3,
    flag_q: float = 0.75,
    def_exec_w: float = 0.60,
    def_obv_w: float = 0.40,
    verbose: bool = True,
) -> pd.DataFrame:
    """Alternative legacy wrapper accepting DataFrame directly."""
    scorer = LateralScorer(
        min_minutes=min_minutes,
        min_matches=min_matches,
        flag_q=flag_q,
        verbose=verbose,
        def_exec_w=def_exec_w,
        def_obv_w=def_obv_w,
    )
    return scorer.score(df=per90_df)
