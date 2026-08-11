"""
Streamlit dashboard: explore historical predictions and try the model
interactively. This is the "demo-able" layer for interviews / resume link.

Run:
    streamlit run src/dashboard.py
"""

from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

PROCESSED_DIR = Path(__file__).resolve().parent.parent / "data" / "processed"
MODEL_DIR = PROCESSED_DIR / "models"
RAW_DIR = Path(__file__).resolve().parent.parent / "data" / "raw"
FORM_WINDOW = 4  # trailing games used for "current form"

FEATURE_COLS = [
    "home_roll_points_for", "home_roll_points_against", "home_roll_point_diff",
    "away_roll_points_for", "away_roll_points_against", "away_roll_point_diff",
    "form_diff",
]

st.set_page_config(page_title="NFL Game Predictor", page_icon="🏈", layout="wide")
st.title("🏈 NFL Game Outcome Predictor")
st.caption(
    "Predicts win probability and point margin from each team's trailing "
    "4-game form. Compared against Vegas closing lines as the honest baseline."
)


@st.cache_data
def load_features():
    path = PROCESSED_DIR / "game_features.parquet"
    if not path.exists():
        return None
    return pd.read_parquet(path)


@st.cache_resource
def load_models():
    win_path = MODEL_DIR / "win_classifier.joblib"
    margin_path = MODEL_DIR / "margin_regressor.joblib"
    if not win_path.exists() or not margin_path.exists():
        return None, None
    return joblib.load(win_path), joblib.load(margin_path)


@st.cache_data
def load_team_current_form():
    """
    For each team, compute their trailing average points for/against
    using their most recently PLAYED games (i.e. their form heading
    into their next game, whenever that is). This is what powers the
    "pick two teams" matchup predictor.
    """
    path = RAW_DIR / "schedules.parquet"
    if not path.exists():
        return None

    schedules = pd.read_parquet(path)
    completed = schedules.dropna(subset=["home_score", "away_score"]).copy()

    home = completed.rename(columns={
        "home_team": "team", "away_team": "opponent",
        "home_score": "points_for", "away_score": "points_against",
    })
    away = completed.rename(columns={
        "away_team": "team", "home_team": "opponent",
        "away_score": "points_for", "home_score": "points_against",
    })
    cols = ["season", "week", "gameday", "team", "opponent", "points_for", "points_against"]
    log = pd.concat([home[cols], away[cols]], ignore_index=True)
    log = log.sort_values(["team", "season", "week"])

    # Exclude Week 18: teams with playoff seeding already locked often rest
    # starters, which would badly skew "current form" for teams like that.
    # This only affects the FORM estimate used for future matchups here —
    # it doesn't touch the training data, since Week 18 games are never
    # used as inputs to predict an earlier game.
    log_for_form = log[log["week"] != 18]

    def last_n_form(group: pd.DataFrame) -> pd.Series:
        recent = group.tail(FORM_WINDOW)
        return pd.Series({
            "points_for": recent["points_for"].mean(),
            "points_against": recent["points_against"].mean(),
            "as_of_season": group["season"].iloc[-1],
            "as_of_week": group["week"].iloc[-1],
            "games_used": len(recent),
        })

    form = log_for_form.groupby("team").apply(last_n_form, include_groups=False).reset_index()
    form["point_diff"] = form["points_for"] - form["points_against"]
    return form.sort_values("team")


df = load_features()
win_model, margin_model = load_models()
team_form = load_team_current_form()

if df is None or win_model is None:
    st.warning(
        "No processed data / trained models found yet. Run these first:\n\n"
        "```\npython -m src.data.ingest\npython -m src.features.build_features\npython -m src.models.train\n```"
    )
    st.stop()

tab1, tab2 = st.tabs(["📊 Historical Games", "🔮 Try a Matchup"])

with tab1:
    seasons = sorted(df["season"].unique(), reverse=True)
    season = st.selectbox("Season", seasons)
    season_df = df[df["season"] == season].sort_values("week")

    season_df = season_df.assign(
        model_home_win_prob=win_model.predict_proba(season_df[FEATURE_COLS])[:, 1],
        model_pred_margin=margin_model.predict(season_df[FEATURE_COLS]),
    )

    display_cols = [
        "week", "home_team", "away_team", "home_score", "away_score",
        "spread_line", "model_pred_margin", "model_home_win_prob",
    ]
    st.dataframe(
        season_df[display_cols].round(2),
        use_container_width=True,
        hide_index=True,
    )

with tab2:
    st.subheader("Pick two teams and predict the matchup")

    if team_form is None or team_form.empty:
        st.warning("No team data found. Run `python -m src.data.ingest` first.")
        st.stop()

    teams = sorted(team_form["team"].unique())

    col1, col2 = st.columns(2)
    with col1:
        home_team = st.selectbox("Home team", teams, index=teams.index("KC") if "KC" in teams else 0)
    with col2:
        away_team = st.selectbox("Away team", teams, index=teams.index("BUF") if "BUF" in teams else 1)

    home_row = team_form[team_form["team"] == home_team].iloc[0]
    away_row = team_form[team_form["team"] == away_team].iloc[0]

    st.caption(
        f"Using each team's last {FORM_WINDOW} played games (excluding Week 18, "
        f"when playoff-bound teams often rest starters) as of season "
        f"{int(home_row['as_of_season'])}. "
        f"Re-run `python -m src.data.ingest` during the season to refresh this."
    )

    c1, c2 = st.columns(2)
    c1.metric(f"{home_team} recent avg (for / against)",
              f"{home_row['points_for']:.1f} / {home_row['points_against']:.1f}")
    c2.metric(f"{away_team} recent avg (for / against)",
              f"{away_row['points_for']:.1f} / {away_row['points_against']:.1f}")

    if st.button("Predict matchup", type="primary"):
        if home_team == away_team:
            st.error("Pick two different teams.")
            st.stop()

        row = pd.DataFrame([{
            "home_roll_points_for": home_row["points_for"],
            "home_roll_points_against": home_row["points_against"],
            "home_roll_point_diff": home_row["point_diff"],
            "away_roll_points_for": away_row["points_for"],
            "away_roll_points_against": away_row["points_against"],
            "away_roll_point_diff": away_row["point_diff"],
            "form_diff": home_row["point_diff"] - away_row["point_diff"],
        }])[FEATURE_COLS]

        win_prob = win_model.predict_proba(row)[0, 1]
        margin = margin_model.predict(row)[0]
        winner = home_team if win_prob >= 0.5 else away_team

        c1, c2, c3 = st.columns(3)
        c1.metric(f"{home_team} win probability", f"{win_prob:.0%}")
        c2.metric("Predicted margin (home - away)", f"{margin:+.1f}")
        c3.metric("Predicted winner", winner)

    with st.expander("Or enter custom stats instead"):
        st.caption("Overrides the auto-looked-up form above.")
        cc1, cc2 = st.columns(2)
        with cc1:
            st.markdown(f"**{home_team} (home)**")
            home_pf = st.number_input("Points scored/game (avg)", value=float(home_row["points_for"]), key="hpf")
            home_pa = st.number_input("Points allowed/game (avg)", value=float(home_row["points_against"]), key="hpa")
        with cc2:
            st.markdown(f"**{away_team} (away)**")
            away_pf = st.number_input("Points scored/game (avg)", value=float(away_row["points_for"]), key="apf")
            away_pa = st.number_input("Points allowed/game (avg)", value=float(away_row["points_against"]), key="apa")

        if st.button("Predict with custom stats"):
            home_diff = home_pf - home_pa
            away_diff = away_pf - away_pa
            row = pd.DataFrame([{
                "home_roll_points_for": home_pf,
                "home_roll_points_against": home_pa,
                "home_roll_point_diff": home_diff,
                "away_roll_points_for": away_pf,
                "away_roll_points_against": away_pa,
                "away_roll_point_diff": away_diff,
                "form_diff": home_diff - away_diff,
            }])[FEATURE_COLS]

            win_prob = win_model.predict_proba(row)[0, 1]
            margin = margin_model.predict(row)[0]

            c1, c2 = st.columns(2)
            c1.metric("Home win probability", f"{win_prob:.0%}")
            c2.metric("Predicted margin (home - away)", f"{margin:+.1f}")