import time
import pandas as pd
from openskill.models import BradleyTerryFull, BradleyTerryFullRating
from json import loads

COMPONENTS = ["auto", "teleop", "endgame"]

TEAM_BLACKLIST = [
    "frc9980",
    "frc9981",
    "frc9982",
    "frc9983",
    "frc9984",
    "frc9985",
    "frc9986",
    "frc9987",
    "frc9988",
    "frc9989",
    "frc9990",
    "frc9991",
    "frc9992",
    "frc9993",
    "frc9994",
    "frc9995",
    "frc9996",
    "frc9997",
    "frc9998",
    "frc9999",
]


def breakdown_match(match, year):

    red_score_breakdown = match["red_score_breakdown"]
    blue_score_breakdown = match["blue_score_breakdown"]

    red_auto = red_score_breakdown.get("autoPoints", 0)
    blue_auto = blue_score_breakdown.get("autoPoints", 0)

    teleop_mapping = {
        2016: ("teleopCrossingPoints", "teleopBoulderPoints"),
        2017: ("teleopFuelPoints", "teleopRotorPoints"),
        2018: ("teleopOwnershipPoints", "vaultPoints"),
        2019: ("cargoPoints", "hatchPanelPoints"),
        2020: ("teleopCellPoints", "controlPanelPoints"),
        2022: ("teleopCargoPoints",),
        2023: ("teleopGamePiecePoints", "linkPoints"),
        2024: ("teleopTotalNotePoints",),
    }

    if year in teleop_mapping:
        points_keys = teleop_mapping[year]
        red_teleop = sum(red_score_breakdown.get(key, 0) for key in points_keys)
        blue_teleop = sum(blue_score_breakdown.get(key, 0) for key in points_keys)

    if year == 2023:
        red_endgame = (
            red_score_breakdown.get("teleopPoints", 0)
            + red_score_breakdown.get("linkPoints", 0)
        ) - red_teleop
        blue_endgame = (
            blue_score_breakdown.get("teleopPoints", 0)
            + blue_score_breakdown.get("linkPoints", 0)
        ) - blue_teleop
    else:
        red_endgame = red_score_breakdown.get("teleopPoints", 0) - red_teleop
        blue_endgame = blue_score_breakdown.get("teleopPoints", 0) - blue_teleop

    return (
        red_auto,
        red_teleop,
        red_endgame,
        blue_auto,
        blue_teleop,
        blue_endgame,
    )


teams_df = pd.read_csv("data/teams.csv")

ratings = {}

model = BradleyTerryFull()

for year in range(2002, 2025):
    try:
        matches_df = pd.read_csv(f"data/matches/{year}_matches.csv")
    except FileNotFoundError:
        print(f"No data found for {year}")
        continue

    matches_df = matches_df.dropna(subset=["red1", "red2", "blue1", "blue2"])

    matches_df = matches_df[
        ~matches_df["red1"].isin(TEAM_BLACKLIST)
        & ~matches_df["red2"].isin(TEAM_BLACKLIST)
        & ~matches_df["red3"].isin(TEAM_BLACKLIST)
        & ~matches_df["blue1"].isin(TEAM_BLACKLIST)
        & ~matches_df["blue2"].isin(TEAM_BLACKLIST)
        & ~matches_df["blue3"].isin(TEAM_BLACKLIST)
    ]

    matches_df = matches_df[
        (matches_df["blue_score"] != -1) & (matches_df["red_score"] != -1)
    ]

    if year >= 2016:
        matches_df["red_score_breakdown"] = matches_df["red_score_breakdown"].apply(
            loads
        )
        matches_df["blue_score_breakdown"] = matches_df["blue_score_breakdown"].apply(
            loads
        )

    for teams in ratings:
        ratings[teams].sigma = 25.0 / 3.0

    baseline_predictions = 0
    correct_predictions = 0
    total_predictions = 0

    for _, match in matches_df.iterrows():
        red_alliance = [match["red1"], match["red2"]]
        if pd.notna(match["red3"]):
            red_alliance.append(match["red3"])

        blue_alliance = [match["blue1"], match["blue2"]]
        if pd.notna(match["blue3"]):
            blue_alliance.append(match["blue3"])

        for team in red_alliance + blue_alliance:
            if team not in ratings:
                ratings[team] = BradleyTerryFullRating(
                    name=str(team), mu=25.0, sigma=25.0 / 3.0
                )

        if year > 2016:
            (
                red_auto,
                red_teleop,
                red_endgame,
                blue_auto,
                blue_teleop,
                blue_endgame,
            ) = breakdown_match(match, year)

            red_score = red_auto + red_teleop + red_endgame
            blue_score = blue_auto + blue_teleop + blue_endgame

        else:
            red_score = match["red_score"]
            blue_score = match["blue_score"]

        winner = (
            "red"
            if red_score > blue_score
            else "blue" if blue_score > red_score else "tie"
        )

        red_ratings = [ratings[team] for team in red_alliance]
        blue_ratings = [ratings[team] for team in blue_alliance]

        red_pred, blue_pred = model.predict_win(
            teams=[
                red_ratings,
                blue_ratings,
            ]
        )

        if red_pred > blue_pred and winner == "red":
            correct_predictions += 1
        elif red_pred < blue_pred and winner == "blue":
            correct_predictions += 1
        elif red_pred == blue_pred and winner == "tie":
            correct_predictions += 1

        total_predictions += 1
        if winner == "red":
            baseline_predictions += 1

        new_red_ratings, new_blue_ratings = model.rate(
            teams=[red_ratings, blue_ratings],
            scores=[red_score, blue_score],
        )

        for i, team in enumerate(red_alliance):
            ratings[team] = new_red_ratings[i]

        for i, team in enumerate(blue_alliance):
            ratings[team] = new_blue_ratings[i]

    print(
        f"TeamRank Accuracy {year}: ({(correct_predictions / total_predictions) * 100:.2f}%)",
        f"Baseline: ({(baseline_predictions / total_predictions) * 100:.2f}%)",
        f"Delta: ({((correct_predictions - baseline_predictions) / total_predictions) * 100:.2f}%)",
    )

sorted_ratings = sorted(ratings.items(), key=lambda x: x[1].ordinal(), reverse=True)

for team, ratings in sorted_ratings[:10]:
    print(f"Team: {team}, Total Ordinal: {ratings.ordinal()}")
