# NFL Game Outcome Predictor

Predicts NFL game outcomes (winner + point spread) using historical play-by-play
and schedule data. Built as an end-to-end pipeline: data ingestion → feature
engineering → model training/evaluation → API → dashboard.

## Why this project

Most "sports prediction" projects stop at a notebook with an accuracy score.
This one is structured like production software: modular pipeline, tested
data logic, a served model (FastAPI), and a small dashboard on top — while
still being honest about a hard problem (NFL games are genuinely hard to
predict; beating a simple baseline is the real bar).

## Project structure

```
nfl-predictor/
├── src/
│   ├── data/         # pulling raw data from nfl_data_py, caching locally
│   ├── features/     # turning raw games/pbp into model-ready feature rows
│   ├── models/        # training, evaluation, baseline comparison
│   └── api/           # FastAPI app serving predictions
├── tests/             # unit tests for data + feature logic
├── notebooks/          # exploration only — not where the "real" logic lives
├── data/raw/           # downloaded data (gitignored)
└── data/processed/    # cleaned feature tables (gitignored)
```

## Setup

```bash
python3 -m venv venv
source venv/bin/activate       # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

> **Known gotcha:** on some systems, `nfl_data_py` fails to install with a
> `ModuleNotFoundError: No module named 'pkg_resources'` error during build.
> Fix with `pip install "setuptools<81"` before installing requirements.

## Usage

```bash
# 1. Pull and cache raw data (schedules + team stats, 2015-2025)
python -m src.data.ingest

# 2. Build the feature table
python -m src.features.build_features

# 3. Train and evaluate models
python -m src.models.train

# 4. Serve predictions via API
uvicorn src.api.main:app --reload

# 5. Run the dashboard
streamlit run src/dashboard.py
```

## Baseline

The model is evaluated against two baselines:
1. **Always predict home team wins** (home field advantage alone)
2. **Vegas closing spread** (the actual bar to beat — this is hard to beat
   and that's fine; the point is to show you understand this is the real
   benchmark, not just "better than a coin flip")

## Data source

[`nfl_data_py`](https://github.com/nflverse/nfl_data_py) — free, no API key,
pulls from the nflverse project (play-by-play, schedules, rosters, betting
lines going back to 1999).

## Status / Roadmap

- [ ] Data ingestion pipeline
- [ ] Feature engineering (rolling team stats, rest days, home/away, etc.)
- [ ] Baseline models (logistic regression)
- [ ] Better models (XGBoost) + evaluation vs. Vegas lines
- [ ] FastAPI serving endpoint
- [ ] Streamlit dashboard
- [ ] Deployed demo link
