"""
Turn raw schedules + weekly stats into a model-ready feature table:
one row per game, features known BEFORE kickoff only.

This is the module where data-leakage bugs are easiest to introduce
(e.g. using a team's final-season stats to predict an early-season
game). Every feature here is computed using only data available up
to, but not including, the game being predicted.

Run directly to rebuild the processed feature table:
    python -m src.features.build_features
"""

from pathlib import Path
import pandas as pd

RAW_DIR = Path(__file__).resolve().parents[2] / "data" / "raw"
PROCESSED_DIR = Path(__file__).resolve().parents[2] / "data" / "processed"

ROLLING_WINDOW = 4  # games of trailing form to average over


def load_raw() -> tuple[pd.DataFrame, pd.DataFrame]:
    schedules = pd.read_parquet(RAW_DIR / "schedules.parquet")
    weekly = pd.read_parquet(RAW_DIR / "weekly_stats.parquet")
    return schedules, weekly


def _team_game_log(schedules: pd.DataFrame) -> pd.DataFrame:
    """
    Reshape schedules (one row per game) into one row per team-game,
    so we can compute each team's trailing form regardless of whether
    they were home or away in a given week.
    """
    home = schedules.rename(columns={
        "home_team": "team", "away_team": "opponent",
        "home_score": "points_for", "away_score": "points_against",
    })
    home["is_home"] = 1

    away = schedules.rename(columns={
        "away_team": "team", "home_team": "opponent",
        "away_score": "points_for", "home_score": "points_against",
    })
    away["is_home"] = 0

    cols = ["season", "week", "game_id", "gameday", "team", "opponent",
            "points_for", "points_against", "is_home"]
    log = pd.concat([home[cols], away[cols]], ignore_index=True)
    log = log.sort_values(["team", "season", "week"])
    return log


def _add_rolling_form(log: pd.DataFrame, window: int = ROLLING_WINDOW) -> pd.DataFrame:
    """
    Trailing average points for/against, shifted by one game so the
    CURRENT game's result is never included in its own features.
    """
    log = log.copy()
    grp = log.groupby("team")

    log["roll_points_for"] = (
        grp["points_for"].transform(lambda s: s.shift(1).rolling(window, min_periods=1).mean())
    )
    log["roll_points_against"] = (
        grp["points_against"].transform(lambda s: s.shift(1).rolling(window, min_periods=1).mean())
    )
    log["roll_point_diff"] = log["roll_points_for"] - log["roll_points_against"]

    # Simple rest-days proxy: games since last game (within season)
    log["games_played_so_far"] = grp.cumcount()

    return log


def build_feature_table(schedules: pd.DataFrame) -> pd.DataFrame:
    """
    Produces one row per game with pre-game features for both teams
    and the target variables (home_win, actual margin).
    """
    completed = schedules.dropna(subset=["home_score", "away_score"]).copy()

    log = _team_game_log(completed)
    log = _add_rolling_form(log)

    home_feats = log[log["is_home"] == 1][
        ["game_id", "roll_points_for", "roll_points_against",
         "roll_point_diff", "games_played_so_far"]
    ].add_prefix("home_")
    home_feats = home_feats.rename(columns={"home_game_id": "game_id"})

    away_feats = log[log["is_home"] == 0][
        ["game_id", "roll_points_for", "roll_points_against",
         "roll_point_diff", "games_played_so_far"]
    ].add_prefix("away_")
    away_feats = away_feats.rename(columns={"away_game_id": "game_id"})

    features = completed.merge(home_feats, on="game_id").merge(away_feats, on="game_id")

    # Target variables
    features["home_win"] = (features["home_score"] > features["away_score"]).astype(int)
    features["actual_margin"] = features["home_score"] - features["away_score"]

    # Feature: form differential between the two teams
    features["form_diff"] = features["home_roll_point_diff"] - features["away_roll_point_diff"]

    # Drop early-season games where rolling stats are mostly NaN/unstable
    features = features[
        (features["home_games_played_so_far"] >= 2) &
        (features["away_games_played_so_far"] >= 2)
    ]

    keep_cols = [
        "game_id", "season", "week", "gameday",
        "home_team", "away_team", "home_score", "away_score",
        "home_roll_points_for", "home_roll_points_against", "home_roll_point_diff",
        "away_roll_points_for", "away_roll_points_against", "away_roll_point_diff",
        "form_diff",
        "spread_line", "total_line",  # Vegas baseline, kept for comparison only
        "home_win", "actual_margin",
    ]
    return features[keep_cols].reset_index(drop=True)


def main() -> None:
    schedules, _ = load_raw()
    feature_table = build_feature_table(schedules)

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    out_path = PROCESSED_DIR / "game_features.parquet"
    feature_table.to_parquet(out_path, index=False)
    print(f"Saved {len(feature_table):,} games with features to {out_path}")


if __name__ == "__main__":
    main()
