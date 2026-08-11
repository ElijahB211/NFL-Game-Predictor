# NFL Game Outcome Predictor

I built this to predict NFL game outcomes (who wins + by how much) using every team's recent form. I structured it as an actual pipeline (pull data, build features, train models, give out predictions) because I wanted it to look and work like something you'd actually maintain and be able to use for a long time.

Live demo:(https://nfl-game-predictor-z937zdjujcfgakshse4yxb.streamlit.app/)
Repo: https://github.com/ElijahB211/NFL-Game-Predictor

## Results

I tested this on 570 games from the 2024 and 2025 seasons that the model never saw during training.

- Guessing the home team wins every time: 54.0% accuracy
- My model: 63.9% accuracy
- Vegas closing lines: 68.3% accuracy

Getting this close to a line from vegas is somewhat an accomplishment from this project. I didn't want to make this be 100% accuracy because I know that would be impossible.

## What's actually in here

Two tabs in the dashboard:

1. A historical games view. You pick any season within the last 10 years, see real games with my model's predictions next to what actually happened and what Vegas had the line at.
2. A "try a matchup" tab which is my plan to be the favorite part of this. Here, you pick two teams and it predicts the outcome using their actual recent scoring form. This can be used to have the edge on friends if you plan on making fun bets.

## How it's organized

```
src/
  data/         - pulls raw data and caches it locally
  features/     - turns raw games into features the model can use
  models/       - trains and evaluates the models
  api/          - a small FastAPI endpoint if you want predictions outside the dashboard
  dashboard.py  - the actual Streamlit app
tests/          - a few unit tests, mostly checking I'm not leaking future data into training
```

## Running it yourself

```
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

python -m src.data.ingest
python -m src.features.build_features
python -m src.models.train
streamlit run src/dashboard.py
```

## Updating it during the season

Unfortunately, nothing auto-refreshes. If you want current data during the season, you rerun the three commands above (ingest, build_features, train) I'd say once a week is reasonable, Thursdays work well since that's usually after that week's stat corrections are finalized. The ingest script pulls whatever years are available automatically and just skips a season/week if it's not published yet instead of crashing.

## Data source

This uses nflreadpy, which is free and pulls from the nflverse project. I originally built this on an older package called nfl_data_py, but that's been officially deprecated, so partway through I migrated everything over to nflreadpy. That actually fixed a real problem I was having — the older package wasn't returning the most recent season's data at all.

## Things this doesn't do well (and I think that's worth saying plainly)

- It has no idea who's actually on a team's roster. It only sees points scored and allowed. If a team got way better in the offseason because of a trade or a draft pick, the model won't know that until a few real games have been played.
- The very start of a new season is the weakest spot. Before a team has played any games yet, I have it fall back to their full average from the previous season instead of a shaky 4 game window from months ago, which helps, but it's still not as good as real current-season data.
- I excluded Week 18 from the "recent form" calculation on purpose, a lot of teams that have already secured their spot in the playoffs rest their starters that week, and including it was making some good teams look mediocre.
- Nothing here updates on its own unfortunately. This isn't a major issue, I just thought too add this in but during the season I will have to manually rerun the pipeline to pull new games.

## Bugs I actually ran into

While testing this, I noticed my "Vegas baseline" accuracy came out to about 29%, which is almost impossible for real professional betting lines. The reason this even happen was because I had the sign backwards on how I was reading the spread column. Also, I noticed the model initially had the Kansas City Chiefs to lose to one of the worst teams which I knew was anohter impossible feature. This is because I did not account for the last week of the season(Week 18) in the last 4 games which was an outliar due to playoff teams resting their starters and not putting up their usual stats. Both were legitimate bugs in my evaluation logic, not just typos, and I only caught them by actually analyzing the results to see if they made sense.

## Built with

Python, pandas, scikit-learn, XGBoost, Streamlit, FastAPI, pytest, nflreadpy, AI assistance for little debug and code review


