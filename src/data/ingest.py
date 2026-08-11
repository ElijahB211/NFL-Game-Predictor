"""
Pull raw NFL data via nfl_data_py and cache it locally as parquet.

Running this module directly re-downloads and re-caches everything:
    python -m src.data.ingest

Years are determined dynamically (2015 through the current calendar year),
so re-running this later in the season — or in a future year — will
automatically try to pull the newest available data. Seasons/weeks that
aren't published yet in the data source are skipped gracefully rather
than crashing the whole run.
"""

from datetime import date
from pathlib import Path

import nfl_data_py as nfl
import pandas as pd

RAW_DIR = Path(__file__).resolve().parents[2] / "data" / "raw"

CURRENT_YEAR = date.today().year
DEFAULT_YEARS = list(range(2015, CURRENT_YEAR + 1))  # e.g. 2015-2026


def fetch_schedules(years: list[int] = DEFAULT_YEARS) -> pd.DataFrame:
    """
    Schedules include final scores, home/away teams, week, and Vegas
    closing lines (spread_line, total_line) — everything needed for
    both the target variable and the baseline comparison. Unlike
    weekly stats, schedules are published for the whole season (with
    scores as NaN for games not yet played), so this rarely 404s.
    """
    df = nfl.import_schedules(years)
    return df


def fetch_team_stats(years: list[int] = DEFAULT_YEARS) -> pd.DataFrame:
    """
    Weekly team-level stats (points, yards, turnovers, etc.) used to
    build rolling pre-game features. Fetched ONE YEAR AT A TIME so
    that a season without published weekly data yet (e.g. the most
    recent one, early in the year) doesn't fail the entire pipeline —
    it's just skipped, with a printed note.
    """
    frames = []
    for year in years:
        try:
            frames.append(nfl.import_weekly_data([year]))
        except Exception as e:
            print(f"  Skipping {year} weekly stats (not available yet): {e}")
    if not frames:
        raise RuntimeError("No weekly stats could be fetched for any year.")
    return pd.concat(frames, ignore_index=True)


def save(df: pd.DataFrame, name: str) -> Path:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    path = RAW_DIR / f"{name}.parquet"
    df.to_parquet(path, index=False)
    print(f"Saved {len(df):,} rows to {path}")
    return path


def main() -> None:
    print(f"Fetching schedules {DEFAULT_YEARS[0]}-{DEFAULT_YEARS[-1]} "
          f"(includes Vegas lines + final scores)...")
    schedules = fetch_schedules()
    save(schedules, "schedules")

    print(f"Fetching weekly team/player stats {DEFAULT_YEARS[0]}-{DEFAULT_YEARS[-1]}...")
    weekly = fetch_team_stats()
    save(weekly, "weekly_stats")

    print("Done. Raw data cached in data/raw/")


if __name__ == "__main__":
    main()