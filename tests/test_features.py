"""
Unit tests for feature engineering — focused on catching data leakage,
since that's the easiest and most damaging bug in this kind of project.
"""

import pandas as pd
import pytest

from src.features.build_features import _team_game_log, _add_rolling_form


@pytest.fixture
def toy_schedule():
    """Three games for team 'A' across three weeks."""
    return pd.DataFrame({
        "season": [2024, 2024, 2024],
        "week": [1, 2, 3],
        "game_id": ["g1", "g2", "g3"],
        "gameday": ["2024-09-08", "2024-09-15", "2024-09-22"],
        "home_team": ["A", "B", "A"],
        "away_team": ["B", "A", "C"],
        "home_score": [20, 14, 30],
        "away_score": [10, 24, 20],
    })


def test_team_game_log_row_count(toy_schedule):
    """Each game should produce exactly two team-game rows (home + away)."""
    log = _team_game_log(toy_schedule)
    assert len(log) == len(toy_schedule) * 2


def test_team_game_log_points_correctly_assigned(toy_schedule):
    log = _team_game_log(toy_schedule)
    team_a_week1 = log[(log["team"] == "A") & (log["week"] == 1)].iloc[0]
    assert team_a_week1["points_for"] == 20
    assert team_a_week1["points_against"] == 10
    assert team_a_week1["is_home"] == 1


def test_rolling_form_excludes_current_game(toy_schedule):
    """
    CRITICAL leakage test: a team's rolling average going INTO a game
    must not include that game's own result.
    """
    log = _team_game_log(toy_schedule)
    log = _add_rolling_form(log, window=4)

    team_a_week1 = log[(log["team"] == "A") & (log["week"] == 1)].iloc[0]
    # No prior games exist yet, so rolling average must be NaN, not 20.
    assert pd.isna(team_a_week1["roll_points_for"])

    team_a_week3 = log[(log["team"] == "A") & (log["week"] == 3)].iloc[0]
    # By week 3, rolling average should reflect weeks 1-2 only (20, 24),
    # not include week 3's own 30 points.
    assert team_a_week3["roll_points_for"] == pytest.approx((20 + 24) / 2)


def test_games_played_counter_starts_at_zero(toy_schedule):
    log = _team_game_log(toy_schedule)
    log = _add_rolling_form(log)
    team_a_week1 = log[(log["team"] == "A") & (log["week"] == 1)].iloc[0]
    assert team_a_week1["games_played_so_far"] == 0
