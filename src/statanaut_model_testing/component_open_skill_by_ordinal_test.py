from random import sample
import pandas as pd
from openskill.models import BradleyTerryFull, BradleyTerryFullRating
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
    elif year == 2024:
        red_teleop = red_score_breakdown["teleopTotalNotePoints"]
        blue_teleop = blue_score_breakdown["teleopTotalNotePoints"]

        red_endgame = red_score_breakdown["teleopPoints"] - red_teleop
        blue_endgame = blue_score_breakdown["teleopPoints"] - blue_teleop

    return red_auto, red_teleop, red_endgame, blue_auto, blue_teleop, blue_endgame


teams_df = pd.read_csv("data/modern_teams.csv")

ratings = {}
ratings_over_years = {}
rookie_year = {
    team_key: year for team_key, year in zip(teams_df["key"], teams_df["rookie_year"])
}
average_ordinal_by_year = []

for team_key in teams_df["key"]:
    ratings[team_key] = {
        "auto": BradleyTerryFullRating(
            name=f"{team_key}_auto", mu=25.0, sigma=25.0 / 3.0
        ),
        "teleop": BradleyTerryFullRating(
            name=f"{team_key}_teleop", mu=25.0, sigma=25.0 / 3.0
        ),
        "endgame": BradleyTerryFullRating(
            name=f"{team_key}_endgame", mu=25.0, sigma=25.0 / 3.0
        ),
    }
    ratings_over_years[team_key] = []

model = BradleyTerryFull()

years_range = list(range(2016, 2025))

for year in years_range:

    try:
        matches_df = pd.read_csv(
            f"data/matches/{year}_matches.csv",
        )

        for team_key in ratings:
            if (
                team_key in matches_df["red1"].values
                or team_key in matches_df["red2"].values
                or team_key in matches_df["red3"].values
                or team_key in matches_df["blue1"].values
                or team_key in matches_df["blue2"].values
                or team_key in matches_df["blue3"].values
            ):
                for component in ["auto", "teleop", "endgame"]:
                    ratings[team_key][component].sigma = 25.0 / 3.0

    except FileNotFoundError:
        print(f"No data found for {year}")
        for team_key in ratings_over_years:
            if ratings_over_years[team_key]:
                ratings_over_years[team_key].append(ratings_over_years[team_key][-1])

        average_ordinal_by_year.append(average_ordinal_by_year[-1])

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

        red_total_ordinal = sum(
            ratings[team][component].ordinal()
            for team in red_alliance
            for component in components
        )

        blue_total_ordinal = sum(
            ratings[team][component].ordinal()
            for team in blue_alliance
            for component in components
        )

        predicted_winner = (
            "red"
            if red_total_ordinal > blue_total_ordinal
            else "blue" if blue_total_ordinal > red_total_ordinal else "tie"
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

            red_ratings = [ratings[team][component] for team in red_alliance]
            blue_ratings = [ratings[team][component] for team in blue_alliance]

            new_red_ratings, new_blue_ratings = model.rate(
                teams=[red_ratings, blue_ratings],
                scores=[component_red_score, component_blue_score],
            )

            for i, team_key in enumerate(red_alliance):
                ratings[team_key][component] = new_red_ratings[i]

            for i, team_key in enumerate(blue_alliance):
                ratings[team_key][component] = new_blue_ratings[i]

    average_ordinal = 0
    num_teams_played = 0

    for team_key in ratings:
        if year >= rookie_year[team_key]:
            overall_ordinal = sum(
                ratings[team_key][component].mu for component in components
            )
            ratings_over_years[team_key].append(overall_ordinal)

            average_ordinal += overall_ordinal
            num_teams_played += 1

    average_ordinal /= num_teams_played
    average_ordinal_by_year.append(average_ordinal)

    print(
        f"TeamRank Accuracy {year}: ({(correct_predictions / len(matches_df)) * 100:.2f}%)",
        f"Baseline: ({(baseline_predictions / len(matches_df)) * 100:.2f}%)",
        f"Delta: ({((correct_predictions - baseline_predictions) / len(matches_df)) * 100:.2f}%)",
    )
    print()

sorted_ratings = sorted(
    ratings.items(),
    key=lambda x: sum(x[1][component].mu for component in components),
    reverse=True,
)

random_sample = sample(sorted_ratings, 4)

teams_to_graph = sorted_ratings[:8] + random_sample + sorted_ratings[-3:]

for team_key, rating_components in teams_to_graph:
    total_ordinal = sum(
        rating_components[component].ordinal() for component in components
    )
    print(
        f"Team: {team_key}, Total Ordinal: {total_ordinal}, Components: Auto: {rating_components['auto'].ordinal()}, Teleop: {rating_components['teleop'].ordinal()}, Endgame: {rating_components['endgame'].ordinal()}"
    )

for team_key, ratings in ratings_over_years.items():
    padded_ratings = [None] * (len(years_range) - len(ratings)) + ratings
    ratings_over_years[team_key] = padded_ratings


plt.figure(figsize=(12, 8))

for team_key, _ in teams_to_graph:
    plt.plot(
        years_range,
        ratings_over_years[team_key],
        marker="o",
        label=f"{teams_df.loc[teams_df['key'] == team_key, 'name'].values[0]} ({teams_df.loc[teams_df['key'] == team_key, 'number'].values[0]})",
    )

plt.plot(
    years_range,
    average_ordinal_by_year,
    color="green",
    linestyle="--",
    label="Average Rating By Year",
)

plt.xlabel("Year")
plt.xlim(years_range[0], years_range[-1])
plt.xticks(years_range, rotation=45)
plt.ylabel("Ordinal Rating (μ - 3σ)")
plt.title("TeamRank Rating Over Time (2022-2024) | Bradley-Terry")

plt.legend(loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=4, fontsize="medium")

plt.subplots_adjust(bottom=0.25)
plt.grid(True, which="both", linestyle="--", linewidth=0.5)
plt.show()
