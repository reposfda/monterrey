
import numpy as np
import pandas as pd

from sklearn.linear_model import LinearRegression

# columnas esperadas
COL_TEAM = "team_name"
COL_PLAYER = "player_name"
COL_PERF = "Overall_Score_Final"
COL_COST = "Cost_Share"


# =========================================================
# MODELO DE PREDICCION
# =========================================================
def fit_market_curve(df: pd.DataFrame, perf_col: str = COL_PERF, cost_col: str = COL_COST):
    df_clean = df.dropna(subset=[perf_col, cost_col]).copy()
    X = df_clean[[perf_col]].values
    y = df_clean[cost_col].values

    model = LinearRegression()
    model.fit(X, y)

    y_pred = model.predict(X)
    residuals = y - y_pred
    residual_std = residuals.std(ddof=1) if len(residuals) > 1 else 0.0

    return model, float(residual_std)


def suggest_salary_range_for_player(
    df: pd.DataFrame,
    player_name: str,
    perf_col: str = COL_PERF,
    cost_col: str = COL_COST,
    player_col: str = COL_PLAYER,
):
    model, resid_std = fit_market_curve(df, perf_col, cost_col)

    row = df.loc[df[player_col] == player_name]
    if row.empty:
        raise ValueError(f"Jugador '{player_name}' no encontrado en el DataFrame.")

    p = float(row[perf_col].iloc[0])
    current_cost_share = float(row[cost_col].iloc[0])

    fair_cost_share = float(model.predict(np.array([[p]]))[0])

    low = max(fair_cost_share - 0.5 * resid_std, 0.0)
    high = max(fair_cost_share + 0.5 * resid_std, 0.0)
    max_reasonable = max(fair_cost_share + 1.0 * resid_std, 0.0)

    return {
        "player_name": player_name,
        "perf": p,
        "current_cost_share": current_cost_share,
        "fair_cost_share": fair_cost_share,
        "negotiation_low": low,
        "negotiation_high": high,
        "max_reasonable_cost_share": max_reasonable,
        "model": model,
        "residual_std": resid_std,
    }


def performance_required_ranges(
    model: LinearRegression,
    start_cs: float = 0.00,
    end_cs: float = 0.10,
    step: float = 0.01,
    clip_min: float = 0.0,
    clip_max: float = 100.0,
) -> pd.DataFrame:
    alpha = float(model.intercept_)
    beta = float(model.coef_[0])

    rows = []
    cs_mins = np.arange(start_cs, end_cs, step)

    for cs_min in cs_mins:
        cs_max = cs_min + step

        # Invertimos la recta: perf = (cost_share - alpha) / beta
        # OJO: beta podría ser negativo -> ordenamos min/max para que tenga sentido en tabla
        if beta == 0:
            perf_a = np.nan
            perf_b = np.nan
        else:
            perf_a = (cs_min - alpha) / beta
            perf_b = (cs_max - alpha) / beta

        perf_min = np.nanmin([perf_a, perf_b])
        perf_max = np.nanmax([perf_a, perf_b])

        if np.isfinite(perf_min):
            perf_min = float(np.clip(perf_min, clip_min, clip_max))
        if np.isfinite(perf_max):
            perf_max = float(np.clip(perf_max, clip_min, clip_max))

        rows.append(
            {
                "cost_share_min": float(cs_min),
                "cost_share_max": float(cs_max),
                "performance_min": perf_min,
                "performance_max": perf_max,
                "cost_share_range_label": f"{cs_min:.2f} – {cs_max:.2f}",
                "performance_range_label": (
                    f"{perf_min:.1f} – {perf_max:.1f}" if np.isfinite(perf_min) and np.isfinite(perf_max) else "—"
                ),
            }
        )

    return pd.DataFrame(rows)
