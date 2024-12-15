import pandas as pd
from openskill.models import PlackettLuceRating, PlackettLuce
import matplotlib.pyplot as plt
import numpy as np

teams_df = pd.read_csv("data/teams.csv")

ratings = {}
ratings_over_years = {}

model = PlackettLuce()

years_range = list(range(2002, 2025))

for year in years_range:
    for team in ratings:
        ratings[team].sigma = 25.0 / 3.0

    try:
        matches_df = pd.read_csv(f"data/matches/{year}_matches.csv")
    except FileNotFoundError:
        print(f"No data found for {year}")
        for team in ratings_over_years:
            ratings_over_years[team].append(ratings_over_years[team][-1])
        continue

    if year < 2005:
        matches_df = matches_df.dropna(subset=["red1", "red2", "blue1", "blue2"])
    else:
        matches_df = matches_df.dropna(
            subset=["red1", "red2", "red3", "blue1", "blue2", "blue3"]
        )

    matches_df = matches_df[
        (matches_df["blue_score"] != -1) & (matches_df["red_score"] != -1)
    ]

    correct_predictions = 0
    baseline_predictions = 0

    for _, match in matches_df.iterrows():
        red_alliance = [
            match["red1"],
            match["red2"],
        ]
        if pd.notna(match["red3"]):
            red_alliance.append(match["red3"])

        blue_alliance = [
            match["blue1"],
            match["blue2"],
        ]
        if pd.notna(match["blue3"]):
            blue_alliance.append(match["blue3"])

        for team in red_alliance + blue_alliance:
            if team not in ratings:
                ratings[team] = PlackettLuceRating(
                    name=str(team), mu=25.0, sigma=25.0 / 3.0
                )
                ratings_over_years[team] = []

        red_score = match["red_score"]
        blue_score = match["blue_score"]

        winner = (
            "red"
            if red_score > blue_score
            else "blue" if blue_score > red_score else "tie"
        )

        if winner == "red":
            baseline_predictions += 1

        if winner == "red":
            ranks = [0, 2]
        elif winner == "blue":
            ranks = [2, 0]
        else:
            ranks = [1, 1]

        red_ratings = [ratings[team] for team in red_alliance]
        blue_ratings = [ratings[team] for team in blue_alliance]

        red_win_pred, blue_win_pred = model.predict_win(
            teams=[red_ratings, blue_ratings]
        )

        if red_win_pred > blue_win_pred and winner == "red":
            correct_predictions += 1
        elif red_win_pred < blue_win_pred and winner == "blue":
            correct_predictions += 1
        elif red_win_pred == blue_win_pred and winner == "tie":
            correct_predictions += 1

        new_red_ratings, new_blue_ratings = model.rate(
            teams=[red_ratings, blue_ratings],
            ranks=ranks,
        )

        for i, team in enumerate(red_alliance):
            ratings[team] = new_red_ratings[i]

        for i, team in enumerate(blue_alliance):
            ratings[team] = new_blue_ratings[i]

    for team in ratings:
        ratings_over_years[team].append(ratings[team])

    print(
        f"TeamRank Accuracy {year}: ({(correct_predictions / len(matches_df)) * 100:.2f}%)",
        f"Baseline: ({(baseline_predictions / len(matches_df)) * 100:.2f}%)",
        f"Delta: ({((correct_predictions - baseline_predictions) / len(matches_df)) * 100:.2f}%)",
    )
    print()

top_teams = sorted(ratings.items(), key=lambda x: x[1].ordinal(), reverse=True)[:10]

for team, rating in top_teams:
    print(f"Team: {team}, Mu: {rating.mu}, Sigma: {rating.sigma}")

for team, ratings in ratings_over_years.items():
    team_ratings = [r.mu for r in ratings_over_years[team]]

    padded_ratings = [None] * (len(years_range) - len(team_ratings)) + team_ratings
    ratings_over_years[team] = padded_ratings

avg_mu_by_year = []
for year_idx in range(len(years_range)):
    yearly_mus = [
        ratings_over_years[team][year_idx]
        for team in ratings_over_years
        if ratings_over_years[team][year_idx] is not None
    ]
    avg_mu_by_year.append(np.mean(yearly_mus) if yearly_mus else None)

plt.figure(figsize=(12, 8))

for team, _ in top_teams:
    plt.plot(
        years_range,
        ratings_over_years[team],
        label=f"{teams_df.loc[teams_df['key'] == team, 'name'].values[0]} ({teams_df.loc[teams_df['key'] == team, 'number'].values[0]})",
    )

plt.axhline(y=25, color="r", linestyle="--", label="Baseline (Mu=25)")
plt.plot(
    years_range,
    avg_mu_by_year,
    color="green",
    linestyle="--",
    label="Average Rating By Year",
)


plt.xlabel("Year")
plt.xlim(2002.0, 2024.0)
plt.xticks(years_range, rotation=45)
plt.ylabel("Rating (Mu)")
plt.title("Top Teams Ratings Over Time")
plt.legend()
plt.grid(True, which="both", linestyle="--", linewidth=0.5)
plt.show()
