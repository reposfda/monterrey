# scoring/base.py
# -*- coding: utf-8 -*-
"""
Base class for position scoring.

Provides:
- Template Method pattern for scoring workflow
- Shared helper functions
- Common filtering and calculation logic
"""
from __future__ import annotations
from abc import ABC, abstractmethod
from pathlib import Path
import numpy as np
import pandas as pd

from .io import read_input
from positions_config import normalize_group, sb_positions_for


class PositionScorer(ABC):
    """
    Abstract base class for scoring players by position.
    
    This class implements the Template Method pattern, where the overall
    scoring workflow is fixed but specific details (categories, weights)
    are defined by subclasses.
    
    Attributes
    ----------
    min_minutes : int
        Minimum minutes played to include player
    min_matches : int
        Minimum matches played to include player
    flag_q : float
        Quantile for flagging top performers (0.75 = top 25%)
    verbose : bool
        Whether to print progress messages
    
    Abstract Properties
    -------------------
    position_group : str
        Name of the position group (e.g., 'Delantero', 'Volante')
    categories : dict
        Definition of scoring categories and their metrics
    category_weights : dict
        Weights for each category in final score
    
    Examples
    --------
    >>> class MyScorer(PositionScorer):
    ...     @property
    ...     def position_group(self):
    ...         return "Delantero"
    ...     
    ...     @property
    ...     def categories(self):
    ...         return {"Score_Goles": [("goals_per90", 1.0, False)]}
    ...     
    ...     @property
    ...     def category_weights(self):
    ...         return {"Score_Goles": 1.0}
    >>> 
    >>> scorer = MyScorer()
    >>> result = scorer.score(df=my_dataframe)
    """
    
    def __init__(
        self,
        min_minutes: int = 450,
        min_matches: int = 3,
        flag_q: float = 0.75,
        verbose: bool = True
    ):
        """
        Initialize PositionScorer.
        
        Parameters
        ----------
        min_minutes : int, default=450
            Minimum minutes played (default ~5 matches)
        min_matches : int, default=3
            Minimum matches played
        flag_q : float, default=0.75
            Quantile threshold for flags (0.75 = top 25%)
        verbose : bool, default=True
            Print progress messages
        """
        self.min_minutes = min_minutes
        self.min_matches = min_matches
        self.flag_q = flag_q
        self.verbose = verbose
    
    # =========================================================================
    # ABSTRACT PROPERTIES (must be defined by subclasses)
    # =========================================================================
    
    @property
    @abstractmethod
    def position_group(self) -> str:
        """
        Name of the position group.
        
        Returns
        -------
        str
            Position name (e.g., 'Delantero', 'Volante', 'Lateral')
        """
        pass
    
    @property
    @abstractmethod
    def categories(self) -> dict:
        """
        Definition of scoring categories.
        
        Returns
        -------
        dict
            Format: {
                'Score_CategoryName': [
                    (metric_name, weight, inverted),
                    ...
                ]
            }
            
            Where:
            - metric_name: str, column name in DataFrame
            - weight: float, weight of this metric in category
            - inverted: bool, True for metrics where lower is better
        
        Examples
        --------
        {
            'Score_Finalizacion': [
                ('goals_per90', 0.5, False),
                ('xg_per_shot', 0.3, False),
                ('turnovers_per90', 0.2, True),  # inverted
            ]
        }
        """
        pass
    
    @property
    @abstractmethod
    def category_weights(self) -> dict:
        """
        Weights for each category in total score.
        
        Returns
        -------
        dict
            Format: {'Score_CategoryName': weight, ...}
            Weights should sum to 1.0
        
        Examples
        --------
        {
            'Score_Finalizacion': 0.4,
            'Score_Pases': 0.3,
            'Score_Defensa': 0.3,
        }
        """
        pass
    
    # =========================================================================
    # SHARED HELPER FUNCTIONS
    # =========================================================================
    
    @staticmethod
    def safe_numeric(s: pd.Series) -> pd.Series:
        """
        Convert series to numeric, coercing errors to NaN.
        
        Parameters
        ----------
        s : pd.Series
            Series to convert
        
        Returns
        -------
        pd.Series
            Numeric series with errors as NaN
        """
        return pd.to_numeric(s, errors="coerce")
    
    @staticmethod
    def pct_rank_0_100(s: pd.Series) -> pd.Series:
        """
        Calculate percentile rank from 0 to 100.
        
        Parameters
        ----------
        s : pd.Series
            Series to rank
        
        Returns
        -------
        pd.Series
            Percentile ranks (0-100), with NaN preserved
        """
        x = s.copy()
        m = x.notna()
        out = pd.Series(np.nan, index=x.index, dtype="float64")
        out.loc[m] = x.loc[m].rank(pct=True, method="average") * 100.0
        return out
    
    @staticmethod
    def wavg(df: pd.DataFrame, cols_weights: list) -> pd.Series:
        """
        Calculate weighted average handling NaN values properly.
        
        Parameters
        ----------
        df : pd.DataFrame
            DataFrame with columns to average
        cols_weights : list
            List of (column_name, weight) tuples
        
        Returns
        -------
        pd.Series
            Weighted average, with NaN where all inputs are NaN
        """
        cols = [c for c, _ in cols_weights if c in df.columns]
        if not cols:
            return pd.Series(np.nan, index=df.index)
        
        w = np.array([w for c, w in cols_weights if c in df.columns], dtype="float64")
        if w.sum() <= 0:
            return pd.Series(np.nan, index=df.index)
        
        w = w / w.sum()
        mat = np.vstack([df[c].to_numpy(dtype="float64") for c in cols]).T
        num = np.nansum(mat * w, axis=1)
        den = np.nansum((~np.isnan(mat)) * w, axis=1)
        return pd.Series(np.where(den > 0, num / den, np.nan), index=df.index)
    
    @staticmethod
    def filter_by_position_group(df: pd.DataFrame, group: str) -> pd.DataFrame:
        """
        Filter DataFrame to only include players in given position group.
        
        Parameters
        ----------
        df : pd.DataFrame
            DataFrame with 'primary_position' column
        group : str
            Position group name
        
        Returns
        -------
        pd.DataFrame
            Filtered DataFrame
        """
        group = normalize_group(group)
        valid_positions = sb_positions_for(group)
        return df[df["primary_position"].isin(valid_positions)].copy()
    
    # =========================================================================
    # MAIN SCORING METHOD (Template Method)
    # =========================================================================
    
    def score(
        self,
        per90_csv: Path | None = None,
        df: pd.DataFrame | None = None,
        out_csv: Path | None = None,
    ) -> pd.DataFrame:
        """
        Calculate scoring for players (main public method).
        
        This implements the Template Method pattern:
        1. Load data
        2. Filter by position
        3. Filter by time requirements
        4. Calculate derived metrics (hook)
        5. Calculate category scores
        6. Calculate total score
        7. Generate flags
        8. Return/save results
        
        Parameters
        ----------
        per90_csv : Path, optional
            Path to per90 CSV file
        df : pd.DataFrame, optional
            DataFrame with per90 data
        out_csv : Path, optional
            Path to save output CSV
        
        Returns
        -------
        pd.DataFrame
            Scored players, sorted by Score_Total descending
        
        Examples
        --------
        >>> scorer = DelanteroScorer()
        >>> result = scorer.score(df=my_df, min_minutes=450)
        >>> result = scorer.score(per90_csv=Path("data.csv"))
        """
        
        # --- 1. Load data ---
        if self.verbose:
            print("=" * 70)
            print(f"SCORING DE {self.position_group.upper()}")
            print("=" * 70)
        
        df_base = read_input(per90_csv=per90_csv, df=df)
        if self.verbose:
            print(f"✓ Total jugadores en input: {len(df_base):,}")
        
        # --- 2. Filter by position ---
        if self.verbose:
            print(f"\n🔍 Filtrando por posición: {self.position_group}")
        
        df_filtered = self.filter_by_position_group(df_base, self.position_group)
        if self.verbose:
            print(f"✓ Jugadores en {self.position_group}: {len(df_filtered):,}")
        
        # --- 3. Filter by time requirements ---
        df_filtered = self._apply_time_filters(df_filtered)
        
        if df_filtered.empty:
            raise ValueError(
                f"No hay jugadores de {self.position_group} que cumplan los filtros "
                f"(min_minutes={self.min_minutes}, min_matches={self.min_matches})."
            )
        
        # --- 4. Rename columns for compatibility ---
        df_filtered = df_filtered.rename(columns={
            "teams": "team_name",
            "matches_played": "matches",
            "total_minutes": "minutes",
        })
        
        # --- 5. Calculate derived metrics (hook method for subclasses) ---
        df_filtered = self.calculate_derived_metrics(df_filtered)
        
        # --- 6. Calculate category scores ---
        df_scored = self._calculate_category_scores(df_filtered)
        
        # --- 7. Calculate total score ---
        df_scored = self._calculate_total_score(df_scored)
        
        # --- 8. Generate flags ---
        df_scored = self._generate_flags(df_scored)
        
        # --- 9. Sort by total score ---
        df_scored = df_scored.sort_values("Score_Total", ascending=False)
        
        # --- 10. Save if requested ---
        if out_csv:
            df_scored.to_csv(out_csv, index=False, encoding="utf-8")
            if self.verbose:
                print(f"\n💾 Guardado en: {out_csv}")
        
        if self.verbose:
            print(f"\n✅ Scoring completado: {len(df_scored):,} jugadores")
            print("=" * 70)
        
        return df_scored
    
    # =========================================================================
    # INTERNAL METHODS
    # =========================================================================
    
    def _apply_time_filters(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply minimum minutes and matches filters."""
        if self.verbose:
            print(f"\n⏱️  Aplicando filtros:")
            print(f"  - Minutos mínimos: {self.min_minutes}")
            print(f"  - Partidos mínimos: {self.min_matches}")
        
        if "total_minutes" in df.columns:
            df = df[df["total_minutes"] >= self.min_minutes].copy()
        if "matches_played" in df.columns:
            df = df[df["matches_played"] >= self.min_matches].copy()
        
        if self.verbose:
            print(f"✓ Jugadores después de filtros: {len(df):,}")
        
        return df
    
    def _calculate_category_scores(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate score for each category."""
        if self.verbose:
            print("\n📊 Calculando scores por categoría...")
        
        result = df.copy()
        
        for cat_name, metrics in self.categories.items():
            # Convert all metrics to numeric
            for metric, _, _ in metrics:
                if metric in result.columns:
                    result[metric] = self.safe_numeric(result[metric])
            
            # Calculate percentiles for each metric
            pct_cols = []
            for metric, weight, inverted in metrics:
                if metric not in result.columns:
                    if self.verbose:
                        print(f"  ⚠️  Métrica '{metric}' no encontrada, saltando...")
                    continue
                
                pct_col = f"{metric}_pct"
                result[pct_col] = self.pct_rank_0_100(result[metric])
                
                # Invert if metric is negative (e.g., turnovers)
                if inverted:
                    result[pct_col] = 100.0 - result[pct_col]
                
                pct_cols.append((pct_col, weight))
            
            # Weighted average of percentiles
            if pct_cols:
                result[cat_name] = self.wavg(result, pct_cols)
                
                if self.verbose:
                    avg = result[cat_name].mean()
                    print(f"  ✓ {cat_name}: promedio = {avg:.1f}")
            else:
                result[cat_name] = np.nan
                if self.verbose:
                    print(f"  ⚠️  {cat_name}: sin métricas disponibles")
        
        return result
    
    def _calculate_total_score(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate total score as weighted average of categories."""
        if self.verbose:
            print("\n🎯 Calculando score total...")
        
        cat_weights = [
            (cat, weight)
            for cat, weight in self.category_weights.items()
            if cat in df.columns
        ]
        
        if not cat_weights:
            raise ValueError("No hay categorías disponibles para calcular Score_Total")
        
        df["Score_Total"] = self.wavg(df, cat_weights)
        
        if self.verbose:
            avg = df["Score_Total"].mean()
            median = df["Score_Total"].median()
            print(f"  ✓ Score_Total: promedio = {avg:.1f}, mediana = {median:.1f}")
        
        return df
    
    def _generate_flags(self, df: pd.DataFrame) -> pd.DataFrame:
        """Generate flags for top performers in each category."""
        if self.verbose:
            print(f"\n🏆 Generando flags (top {self.flag_q*100:.0f}%)...")
        
        # Flag for total score
        if "Score_Total" in df.columns:
            threshold = df["Score_Total"].quantile(self.flag_q)
            df["flag_Total"] = (df["Score_Total"] >= threshold).astype(int)
            
            if self.verbose:
                count = df["flag_Total"].sum()
                print(f"  ✓ flag_Total: {count} jugadores")
        
        # Flags for each category
        for cat in self.categories.keys():
            if cat in df.columns:
                flag_col = f"flag_{cat}"
                cat_threshold = df[cat].quantile(self.flag_q)
                df[flag_col] = (df[cat] >= cat_threshold).astype(int)
                
                if self.verbose:
                    count = df[flag_col].sum()
                    print(f"  ✓ {flag_col}: {count} jugadores")
        
        return df
    
    # =========================================================================
    # HOOK METHODS (can be overridden by subclasses)
    # =========================================================================
    
    def calculate_derived_metrics(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Hook method to calculate derived metrics specific to position.
        
        Override this method in subclasses if you need to calculate
        metrics that don't exist in the base DataFrame (e.g., ratios,
        combinations, fallbacks).
        
        Parameters
        ----------
        df : pd.DataFrame
            Filtered DataFrame
        
        Returns
        -------
        pd.DataFrame
            DataFrame with derived metrics added
        
        Examples
        --------
        >>> def calculate_derived_metrics(self, df):
        ...     # Calculate xG per shot if missing
        ...     if 'xg_per_shot' not in df.columns:
        ...         df['xg_per_shot'] = df['xg_per90'] / df['shots_per90']
        ...     return df
        """
        if self.verbose:
            print("\n🔧 Calculando métricas derivadas...")
            print("  (ninguna específica de posición)")
        return df
