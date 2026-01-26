# extremos_obv_lanes_builder.py
# Basado en extremos_obv_carriles_iniciopase.py (refactor a funciones)
# Genera métricas per90 de OBV por carriles (origen y origen→destino) para extremos.

import os
import json
import ast
import numpy as np
import pandas as pd

# -------------------------
# Helpers (idénticos a tu script)
# -------------------------
ASSUME_END_CAP = 120

WINGER_POSITIONS = {
    "Right Wing", "Left Wing",
    "Right Midfield", "Left Midfield",
    "RW", "LW", "RM", "LM"
}

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

def get_player_id_series(df: pd.DataFrame) -> pd.Series:
    for cand in ["player_id", "playerId", "idPlayer"]:
        if cand in df.columns:
            return pd.to_numeric(df[cand], errors="coerce")
    if "player" in df.columns:
        obj = df["player"].apply(_coerce_literal)
        return obj.apply(lambda v: v.get("id") if isinstance(v, dict) else np.nan)
    return pd.Series(np.nan, index=df.index)

def get_player_name_series(df: pd.DataFrame) -> pd.Series:
    if "player" in df.columns:
        obj = df["player"].apply(_coerce_literal)
        return obj.apply(lambda v: v.get("name") if isinstance(v, dict) else (str(v) if v is not None else ""))
    for cand in ["player.name", "player_name", "playerName"]:
        if cand in df.columns:
            return df[cand].astype(str)
    return pd.Series("", index=df.index)

def get_position_str(val):
    if pd.isna(val):
        return None
    v = _coerce_literal(val)
    if isinstance(v, dict):
        return v.get("name") or v.get("label")
    return str(v)

def match_end_minute(df_match: pd.DataFrame) -> float:
    mx = pd.to_numeric(df_match.get("minute", pd.Series([], dtype=float)), errors="coerce").max()
    if pd.isna(mx):
        return 90.0
    return float(min(ASSUME_END_CAP, max(90.0, mx)))

def lineup_players_from_match(df_match: pd.DataFrame):
    out = {}
    xi = df_match.loc[df_match["type"].astype(str) == "Starting XI"]
    if xi.empty or "tactics" not in df_match.columns:
        return out
    for tac in xi["tactics"].dropna().tolist():
        tac_obj = _coerce_literal(tac)
        if isinstance(tac_obj, dict) and "lineup" in tac_obj:
            for p in tac_obj.get("lineup", []):
                pid = (p.get("player") or {}).get("id")
                pos = (p.get("position") or {}).get("name") or p.get("position")
                if pid is not None and pos:
                    out[int(pid)] = str(pos)
    return out

def sub_on_minute(df_match: pd.DataFrame, player_id: int):
    subs = df_match.loc[df_match["type"].astype(str) == "Substitution"]
    if subs.empty or "substitution_replacement" not in df_match.columns:
        return None
    mins = []
    for _, row in subs.iterrows():
        repl = _coerce_literal(row["substitution_replacement"])
        if isinstance(repl, dict) and repl.get("id") == player_id:
            m = pd.to_numeric(row.get("minute", np.nan), errors="coerce")
            if pd.notna(m):
                mins.append(float(m))
    return min(mins) if mins else None

def sub_off_minute(df_match: pd.DataFrame, player_id: int):
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

def minutes_played_in_match(df_match: pd.DataFrame, player_id: int, xi_players: dict) -> int:
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

def infer_pitch_width(df: pd.DataFrame) -> float:
    y = pd.to_numeric(df.get("y", np.nan), errors="coerce")
    if not y.notna().any():
        return 80.0
    ymax = y.dropna().quantile(0.995)
    if ymax <= 82:
        return 80.0
    if ymax <= 102:
        return 100.0
    return float(min(120.0, max(80.0, y.dropna().max())))

def lane_id_from_y(y, width):
    if pd.isna(y):
        return np.nan
    y = float(y)
    y = max(0.0, min(width, y))
    step = width / 5.0
    lane = int(y // step) + 1
    return min(5, max(1, lane))

def get_end_y_from_pass_end_location(s):
    v = _coerce_literal(s)
    if isinstance(v, (list, tuple)) and len(v) >= 2:
        try:
            return float(v[1])
        except Exception:
            return np.nan
    return np.nan

def lane_group_2(lane):
    if pd.isna(lane):
        return np.nan
    lane = int(lane)
    return "ext" if lane in (1, 5) else "int"


# -------------------------
# Builder principal
# -------------------------
def build_extremos_obv_lanes_metrics(
    season_label: str,
    path_csv: str,
    minutes_threshold: int = 450,
    min_share_role: float = 0.60,
) -> pd.DataFrame:

    df = pd.read_csv(path_csv, low_memory=False)
    df["_season"] = season_label

    if "match_id" not in df.columns:
        raise ValueError(f"[{season_label}] No existe match_id en el dataset.")

    df["minute"] = pd.to_numeric(df.get("minute", np.nan), errors="coerce")
    df["type"] = df["type"].astype(str)

    df["_player_id"] = get_player_id_series(df)
    df["_player_name"] = get_player_name_series(df)

    df["y"] = pd.to_numeric(df.get("y", np.nan), errors="coerce")
    df["obv_total_net"] = pd.to_numeric(df.get("obv_total_net", np.nan), errors="coerce")

    pitch_width = infer_pitch_width(df)

    # ---- A) minutos por match + posición
    rows = []
    name_by_id = {}

    for mid, g in df.groupby("match_id", sort=False):
        g = g.copy()
        xi_pos = lineup_players_from_match(g)

        participants = set(xi_pos.keys())

        subs = g.loc[g["type"] == "Substitution"]
        if not subs.empty:
            ids_out = get_player_id_series(subs).dropna().astype(int).tolist()
            participants.update(ids_out)

            if "substitution_replacement" in subs.columns:
                for repl in subs["substitution_replacement"].dropna().tolist():
                    obj = _coerce_literal(repl)
                    if isinstance(obj, dict) and obj.get("id") is not None:
                        participants.add(int(obj["id"]))

        ids_ser = get_player_id_series(g)
        nms_ser = get_player_name_series(g)
        tmp = pd.DataFrame({"pid": ids_ser, "pname": nms_ser}).dropna()
        for _, r in tmp.iterrows():
            try:
                name_by_id[int(r["pid"])] = str(r["pname"])
            except Exception:
                pass

        pos_evt = None
        if "position" in g.columns:
            pos_evt = g["position"].apply(get_position_str)

        for pid in participants:
            pid = int(pid)
            mins = minutes_played_in_match(g, pid, xi_pos)
            if mins <= 0:
                continue

            pos_name = xi_pos.get(pid)
            if (pos_name is None) and (pos_evt is not None):
                maskp = (ids_ser == pid)
                pv = pos_evt.loc[maskp].dropna()
                if not pv.empty:
                    pos_name = pv.value_counts().idxmax()

            rows.append({"match_id": mid, "player_id": pid, "minutes": mins, "pos_name": pos_name})

    df_min = pd.DataFrame(rows)
    if df_min.empty:
        raise ValueError(f"[{season_label}] No pude calcular minutos por match.")

    df_min["pos_name"] = df_min["pos_name"].astype(str)
    df_min["is_winger"] = df_min["pos_name"].isin(WINGER_POSITIONS)

    min_total = df_min.groupby("player_id", as_index=False)["minutes"].sum().rename(columns={"minutes":"minutes_total"})
    min_wing  = df_min[df_min["is_winger"]].groupby("player_id", as_index=False)["minutes"].sum().rename(columns={"minutes":"minutes_winger"})

    pool = min_total.merge(min_wing, on="player_id", how="left").fillna({"minutes_winger": 0})
    pool["share_winger"] = pool["minutes_winger"] / pool["minutes_total"].replace({0: np.nan})
    pool["player_name"] = pool["player_id"].map(lambda x: name_by_id.get(int(x), f"player_{int(x)}"))

    pool_wingers = pool[
        (pool["minutes_winger"] >= minutes_threshold) &
        (pool["share_winger"] >= min_share_role)
    ].copy()

    if pool_wingers.empty:
        return pd.DataFrame()

    winger_ids = set(pool_wingers["player_id"].astype(int).tolist())

    # ---- B) pases de extremos
    df_pass = df[(df["_player_id"].isin(winger_ids)) & (df["type"] == "Pass")].copy()
    if df_pass.empty:
        return pd.DataFrame()

    if "pass_end_location" not in df_pass.columns:
        raise ValueError(f"[{season_label}] Falta pass_end_location.")

    df_pass["obv_pass"] = pd.to_numeric(df_pass["obv_total_net"], errors="coerce").fillna(0.0)

    df_pass["start_y"] = pd.to_numeric(df_pass["y"], errors="coerce")
    df_pass["end_y"] = df_pass["pass_end_location"].apply(get_end_y_from_pass_end_location)

    df_pass = df_pass.dropna(subset=["start_y", "end_y"]).copy()

    df_pass["start_lane"] = df_pass["start_y"].apply(lambda yy: lane_id_from_y(yy, pitch_width))
    df_pass["end_lane"]   = df_pass["end_y"].apply(lambda yy: lane_id_from_y(yy, pitch_width))

    df_pass["start_group"] = df_pass["start_lane"].apply(lane_group_2)
    df_pass["end_group"]   = df_pass["end_lane"].apply(lane_group_2)

    # ---- C) agregación 2x2 (ext/int)
    pivot = df_pass.pivot_table(
        index="_player_id",
        columns=["start_group", "end_group"],
        values="obv_pass",
        aggfunc="sum",
        fill_value=0.0
    )

    mi = pd.MultiIndex.from_product([["ext", "int"], ["ext", "int"]], names=["start_group", "end_group"])
    pivot = pivot.reindex(columns=mi, fill_value=0.0)

    pivot.columns = [f"obv_{sg}_to_{eg}_total" for sg, eg in pivot.columns]
    pivot = pivot.reset_index().rename(columns={"_player_id": "player_id"})
    pivot["player_id"] = pivot["player_id"].astype(int)

    pivot["obv_from_ext_total"] = pivot["obv_ext_to_ext_total"] + pivot["obv_ext_to_int_total"]
    pivot["obv_from_int_total"] = pivot["obv_int_to_ext_total"] + pivot["obv_int_to_int_total"]
    pivot["obv_total_pass"] = pivot["obv_from_ext_total"] + pivot["obv_from_int_total"]

    # ---- D) merge + per90
    out = pivot.merge(
        pool_wingers[["player_id","player_name","minutes_winger","minutes_total","share_winger"]],
        on="player_id",
        how="left"
    )

    m = out["minutes_winger"].replace({0: np.nan})

    for col in [
        "obv_ext_to_ext_total","obv_ext_to_int_total","obv_int_to_ext_total","obv_int_to_int_total",
        "obv_from_ext_total","obv_from_int_total","obv_total_pass"
    ]:
        out[col.replace("_total", "_per90")] = out[col] / m * 90.0

    eps = 1e-9
    den = (out["obv_from_int_per90"].fillna(0) + out["obv_from_ext_per90"].fillna(0)).abs() + eps
    out["lane_bias_origin_index"] = (out["obv_from_int_per90"].fillna(0) - out["obv_from_ext_per90"].fillna(0)) / den

    out["_season"] = season_label
    return out.sort_values("obv_total_pass_per90", ascending=False).reset_index(drop=True)


def build_all_seasons(PATHS: dict, out_dir: str, **kwargs) -> pd.DataFrame:
    os.makedirs(out_dir, exist_ok=True)
    all_out = []

    for season, p in PATHS.items():
        res = build_extremos_obv_lanes_metrics(season, p, **kwargs)
        if res.empty:
            continue

        out_path = os.path.join(out_dir, f"extremos_obv_origen_destino_{season.replace('-','')}.csv")
        res.to_csv(out_path, index=False, encoding="utf-8-sig")
        all_out.append(res)

    if all_out:
        df_all = pd.concat(all_out, ignore_index=True)
        out_path_all = os.path.join(out_dir, "extremos_obv_origen_destino_ALL_seasons_stacked.csv")
        df_all.to_csv(out_path_all, index=False, encoding="utf-8-sig")
        return df_all

    return pd.DataFrame()

if __name__ == "__main__":

    PATHS = {"data/events_2024_2025.csv"}

    OUT_DIR = "/outputs"

    build_all_seasons(
        PATHS=PATHS,
        out_dir=OUT_DIR,
        minutes_threshold=450,
        min_share_role=0.60,
    )
