import pandas as pd
from openskill.models import PlackettLuceRating, PlackettLuce
import matplotlib.pyplot as plt
import numpy as np
from json import loads


def breakdown_match(match, year):
    red_score_breakdown = loads(match["red_score_breakdown"])
    blue_score_breakdown = loads(match["blue_score_breakdown"])

    red_auto = red_score_breakdown["autoPoints"]

    blue_auto = blue_score_breakdown["autoPoints"]

    if year == 2016:
        red_teleop = (
            red_score_breakdown["teleopCrossingPoints"]
            + red_score_breakdown["teleopBoulderPoints"]
        )
        blue_teleop = (
            blue_score_breakdown["teleopCrossingPoints"]
            + blue_score_breakdown["teleopBoulderPoints"]
        )

        red_endgame = red_score_breakdown.get("teleopPoints", 0) - red_teleop
        blue_endgame = blue_score_breakdown.get("teleopPoints", 0) - blue_teleop
    elif year == 2017:
        red_teleop = (
            red_score_breakdown["teleopFuelPoints"]
            + red_score_breakdown["teleopRotorPoints"]
        )
        blue_teleop = (
            blue_score_breakdown["teleopFuelPoints"]
            + blue_score_breakdown["teleopRotorPoints"]
        )

        red_endgame = red_score_breakdown["teleopPoints"] - red_teleop
        blue_endgame = blue_score_breakdown["teleopPoints"] - blue_teleop
    elif year == 2018:
        red_teleop = (
            red_score_breakdown["teleopOwnershipPoints"]
            + red_score_breakdown["vaultPoints"]
        )
        blue_teleop = (
            blue_score_breakdown["teleopOwnershipPoints"]
            + blue_score_breakdown["vaultPoints"]
        )

        red_endgame = red_score_breakdown["teleopPoints"] - red_teleop
        blue_endgame = blue_score_breakdown["teleopPoints"] - blue_teleop
    elif year == 2019:
        red_teleop = (
            red_score_breakdown["cargoPoints"] + red_score_breakdown["hatchPanelPoints"]
        )
        blue_teleop = (
            blue_score_breakdown["cargoPoints"]
            + blue_score_breakdown["hatchPanelPoints"]
        )

        red_endgame = red_score_breakdown["teleopPoints"] - red_teleop
        blue_endgame = blue_score_breakdown["teleopPoints"] - blue_teleop
    elif year == 2020:
        red_teleop = (
            red_score_breakdown["teleopCellPoints"]
            + red_score_breakdown["controlPanelPoints"]
        )
        blue_teleop = (
            blue_score_breakdown["teleopCellPoints"]
            + blue_score_breakdown["controlPanelPoints"]
        )

        red_endgame = red_score_breakdown["teleopPoints"] - red_teleop
        blue_endgame = blue_score_breakdown["teleopPoints"] - blue_teleop
    elif year == 2022:
        red_teleop = red_score_breakdown["teleopCargoPoints"]
        blue_teleop = blue_score_breakdown["teleopCargoPoints"]

        red_endgame = red_score_breakdown["teleopPoints"] - red_teleop
        blue_endgame = blue_score_breakdown["teleopPoints"] - blue_teleop
    elif year == 2023:
        red_teleop = red_score_breakdown["teleopGamePiecePoints"]
        blue_teleop = blue_score_breakdown["teleopGamePiecePoints"]

        red_endgame = red_score_breakdown["teleopPoints"] - red_teleop
        blue_endgame = blue_score_breakdown["teleopPoints"] - blue_teleop
    else:
        red_teleop = red_score_breakdown["teleopTotalNotePoints"]
        blue_teleop = blue_score_breakdown["teleopTotalNotePoints"]

        red_endgame = red_score_breakdown["teleopPoints"] - red_teleop
        blue_endgame = blue_score_breakdown["teleopPoints"] - blue_teleop

    return red_auto, red_teleop, red_endgame, blue_auto, blue_teleop, blue_endgame


teams_df = pd.read_csv("data/teams.csv")

ratings = {}
ratings_over_years = {}

for team in teams_df["key"]:
    ratings[team] = {
        "auto": PlackettLuceRating(name=f"{team}_auto", mu=25.0, sigma=25.0 / 3.0),
        "teleop": PlackettLuceRating(name=f"{team}_teleop", mu=25.0, sigma=25.0 / 3.0),
        "endgame": PlackettLuceRating(
            name=f"{team}_endgame", mu=25.0, sigma=25.0 / 3.0
        ),
    }
    ratings_over_years[team] = []

model = PlackettLuce()

years_range = list(range(2016, 2025))

for year in years_range:
    for team in ratings:
        for component in ["auto", "teleop", "endgame"]:
            ratings[team][component].sigma = 25.0 / 2.5

    try:
        matches_df = pd.read_csv(
            f"data/matches/{year}_matches.csv",
        )
    except FileNotFoundError:
        print(f"No data found for {year}")
        for team in ratings_over_years:
            ratings_over_years[team].append(ratings_over_years[team][-1])
        continue

    matches_df = matches_df.dropna(
        subset=[
            "red1",
            "red2",
            "red3",
            "blue1",
            "blue2",
            "blue3",
            "red_score",
            "blue_score",
            "red_score_breakdown",
            "blue_score_breakdown",
        ]
    )

    matches_df = matches_df[
        (matches_df["blue_score"] != -1) & (matches_df["red_score"] != -1)
    ]

    matches_df = matches_df[
        matches_df.apply(
            lambda x: "autoPoints" in loads(x["red_score_breakdown"])
            and "teleopPoints" in loads(x["red_score_breakdown"])
            and "autoPoints" in loads(x["blue_score_breakdown"])
            and "teleopPoints" in loads(x["blue_score_breakdown"]),
            axis=1,
        )
    ]

    correct_predictions = 0
    baseline_predictions = 0

    for _, match in matches_df.iterrows():
        red_alliance = [match["red1"], match["red2"]]
        if pd.notna(match["red3"]):
            red_alliance.append(match["red3"])

        blue_alliance = [match["blue1"], match["blue2"]]
        if pd.notna(match["blue3"]):
            blue_alliance.append(match["blue3"])

        red_auto, red_teleop, red_endgame, blue_auto, blue_teleop, blue_endgame = (
            breakdown_match(match, year)
        )

        total_red_score = match["red_score"]
        total_blue_score = match["blue_score"]

        total_winner = (
            "red"
            if total_red_score > total_blue_score
            else "blue" if total_blue_score > total_red_score else "tie"
        )

        if total_winner == "red":
            baseline_predictions += 1

        components = ["auto", "teleop", "endgame"]

        red_total_mu = sum(
            ratings[team][component].mu
            for team in red_alliance
            for component in components
        )

        blue_total_mu = sum(
            ratings[team][component].mu
            for team in blue_alliance
            for component in components
        )

        predicted_winner = (
            "red"
            if red_total_mu > blue_total_mu
            else "blue" if blue_total_mu > red_total_mu else "tie"
        )

        if predicted_winner == total_winner:
            correct_predictions += 1

        for component in components:

            if component == "auto":
                component_red_score = red_auto
                component_blue_score = blue_auto

            elif component == "teleop":
                component_red_score = red_teleop
                component_blue_score = blue_teleop

            else:
                component_red_score = red_endgame
                component_blue_score = blue_endgame

            component_winner = (
                "red"
                if component_red_score > component_blue_score
                else "blue" if component_blue_score > component_red_score else "tie"
            )

            if component_winner == "red":
                component_ranks = [0, 2]
            elif component_winner == "blue":
                component_ranks = [2, 0]
            else:
                component_ranks = [1, 1]

            red_ratings = [ratings[team][component] for team in red_alliance]
            blue_ratings = [ratings[team][component] for team in blue_alliance]

            new_red_ratings, new_blue_ratings = model.rate(
                teams=[red_ratings, blue_ratings],
                ranks=component_ranks,
            )

            for i, team in enumerate(red_alliance):
                ratings[team][component] = new_red_ratings[i]

            for i, team in enumerate(blue_alliance):
                ratings[team][component] = new_blue_ratings[i]

    for team in ratings:
        overall_mu = sum(ratings[team][component].mu for component in components)
        ratings_over_years[team].append(overall_mu)

    print(
        f"TeamRank Accuracy {year}: ({(correct_predictions / len(matches_df)) * 100:.2f}%)",
        f"Baseline: ({(baseline_predictions / len(matches_df)) * 100:.2f}%)",
        f"Delta: ({((correct_predictions - baseline_predictions) / len(matches_df)) * 100:.2f}%)",
    )
    print()

top_teams = sorted(
    ratings.items(),
    key=lambda x: sum(x[1][component].mu for component in components),
    reverse=True,
)[:10]

for team, rating_components in top_teams:
    total_mu = sum(rating_components[component].mu for component in components)
    print(
        f"Team: {team}, Total Mu: {total_mu}, Components: Auto: {rating_components['auto'].mu}, Teleop: {rating_components['teleop'].mu}, Endgame: {rating_components['endgame'].mu}"
    )

for team, ratings in ratings_over_years.items():
    padded_ratings = [None] * (len(years_range) - len(ratings)) + ratings
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
plt.xlim(years_range[0], years_range[-1])
plt.xticks(years_range, rotation=45)
plt.ylabel("Rating (Mu)")
plt.title("Top Teams Ratings Over Time")
plt.legend()
plt.grid(True, which="both", linestyle="--", linewidth=0.5)
plt.show()