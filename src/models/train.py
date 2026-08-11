"""
Train and evaluate models on the feature table, using a TIME-BASED
split (train on earlier seasons, test on later ones). A random split
would leak information — team strength in 2024 is correlated across
weeks, so shuffling lets the model "see the future."

Compares against two baselines:
    1. Home team always wins
    2. Vegas closing spread (the real bar to beat)

Run directly:
    python -m src.models.train
"""

from pathlib import Path
import joblib
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, log_loss, mean_absolute_error
from xgboost import XGBRegressor

PROCESSED_DIR = Path(__file__).resolve().parents[2] / "data" / "processed"
MODEL_DIR = Path(__file__).resolve().parents[2] / "data" / "processed" / "models"

FEATURE_COLS = [
    "home_roll_points_for", "home_roll_points_against", "home_roll_point_diff",
    "away_roll_points_for", "away_roll_points_against", "away_roll_point_diff",
    "form_diff",
]

TEST_SEASONS = {2024, 2025}  # hold out most recent seasons as test set


def load_features() -> pd.DataFrame:
    return pd.read_parquet(PROCESSED_DIR / "game_features.parquet")


def time_split(df: pd.DataFrame):
    train = df[~df["season"].isin(TEST_SEASONS)].dropna(subset=FEATURE_COLS)
    test = df[df["season"].isin(TEST_SEASONS)].dropna(subset=FEATURE_COLS)
    return train, test


def evaluate_baselines(test: pd.DataFrame) -> dict:
    results = {}

    # Baseline 1: home team always wins
    home_always = (test["home_win"] == 1).mean()
    results["baseline_home_always_wins_acc"] = round(home_always, 4)

    # Baseline 2: Vegas spread implies the favorite
    # spread_line is from the home team's perspective (negative = home favored)
    vegas_valid = test.dropna(subset=["spread_line"])
    vegas_pred = (vegas_valid["spread_line"] > 0).astype(int)
    results["baseline_vegas_acc"] = round(accuracy_score(vegas_valid["home_win"], vegas_pred), 4)
    results["baseline_vegas_mae_margin"] = round(
        mean_absolute_error(vegas_valid["actual_margin"], vegas_valid["spread_line"]), 3
    )

    return results


def train_win_classifier(train: pd.DataFrame, test: pd.DataFrame) -> dict:
    X_train, y_train = train[FEATURE_COLS], train["home_win"]
    X_test, y_test = test[FEATURE_COLS], test["home_win"]

    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    probs = model.predict_proba(X_test)[:, 1]

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, MODEL_DIR / "win_classifier.joblib")

    return {
        "model_win_acc": round(accuracy_score(y_test, preds), 4),
        "model_win_logloss": round(log_loss(y_test, probs), 4),
    }


def train_margin_regressor(train: pd.DataFrame, test: pd.DataFrame) -> dict:
    X_train, y_train = train[FEATURE_COLS], train["actual_margin"]
    X_test, y_test = test[FEATURE_COLS], test["actual_margin"]

    model = XGBRegressor(n_estimators=200, max_depth=3, learning_rate=0.05, random_state=42)
    model.fit(X_train, y_train)

    preds = model.predict(X_test)

    joblib.dump(model, MODEL_DIR / "margin_regressor.joblib")

    return {"model_margin_mae": round(mean_absolute_error(y_test, preds), 3)}


def main() -> None:
    df = load_features()
    train, test = time_split(df)
    print(f"Train: {len(train):,} games ({train['season'].min()}-{train['season'].max()})")
    print(f"Test:  {len(test):,} games ({sorted(TEST_SEASONS)})")

    results = {}
    results.update(evaluate_baselines(test))
    results.update(train_win_classifier(train, test))
    results.update(train_margin_regressor(train, test))

    print("\n--- Results ---")
    for k, v in results.items():
        print(f"{k}: {v}")

    print("\nInterpretation:")
    print("- Beating 'home always wins' is the minimum bar.")
    print("- Getting CLOSE to the Vegas baseline (not necessarily beating it)")
    print("  is a legitimate, honest result — Vegas lines are very hard to beat")
    print("  and saying so shows you understand the problem, not just the code.")


if __name__ == "__main__":
    main()
