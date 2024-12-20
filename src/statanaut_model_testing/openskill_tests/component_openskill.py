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

years_range = list(range(2016, 2025))

model = BradleyTerryFull()

for year in years_range:
    try:
        matches_df = pd.read_csv(f"data/matches/{year}_matches.csv")
    except FileNotFoundError:
        print(f"No data found for {year}")
        continue

    matches_df = matches_df.dropna(
        subset=["red1", "red2", "red3", "blue1", "blue2", "blue3"]
    )

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

    matches_df["red_score_breakdown"] = matches_df["red_score_breakdown"].apply(loads)
    matches_df["blue_score_breakdown"] = matches_df["blue_score_breakdown"].apply(loads)

    for teams in ratings:
        for component in COMPONENTS:
            ratings[teams][component].sigma = 25.0 / 3.0

    baseline_predictions = 0
    correct_predictions = 0
    total_predictions = 0

    for _, match in matches_df.iterrows():
        red_alliance = [match["red1"], match["red2"], match["red3"]]
        blue_alliance = [match["blue1"], match["blue2"], match["blue3"]]

        for team in red_alliance + blue_alliance:
            if team not in ratings:
                ratings[team] = {
                    component: BradleyTerryFullRating(
                        name=f"{team}_{component}", mu=25.0, sigma=25.0 / 3.0
                    )
                    for component in COMPONENTS
                }

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

        winner = (
            "red"
            if red_score > blue_score
            else "blue" if blue_score > red_score else "tie"
        )

        red_pred = 0
        blue_pred = 0
        for component in COMPONENTS:
            pred = model.predict_win(
                teams=[
                    [ratings[team][component] for team in red_alliance],
                    [ratings[team][component] for team in blue_alliance],
                ]
            )

            red_pred += pred[0]
            blue_pred += pred[1]

        if red_pred > blue_pred and winner == "red":
            correct_predictions += 1
        elif red_pred < blue_pred and winner == "blue":
            correct_predictions += 1
        elif red_pred == blue_pred and winner == "tie":
            correct_predictions += 1

        total_predictions += 1
        if winner == "red":
            baseline_predictions += 1

        for component in COMPONENTS:
            if component == "auto":
                component_red_score = red_auto
                component_blue_score = blue_auto

            elif component == "teleop":
                component_red_score = red_teleop
                component_blue_score = blue_teleop

            elif component == "endgame":
                component_red_score = red_endgame
                component_blue_score = blue_endgame

            red_component_ratings = [ratings[team][component] for team in red_alliance]
            blue_component_ratings = [
                ratings[team][component] for team in blue_alliance
            ]

            new_red_component_ratings, new_blue_component_ratings = model.rate(
                teams=[red_component_ratings, blue_component_ratings],
                scores=[component_red_score, component_blue_score],
            )

            for i, team in enumerate(red_alliance):
                ratings[team][component] = new_red_component_ratings[i]

            for i, team in enumerate(blue_alliance):
                ratings[team][component] = new_blue_component_ratings[i]

    print(
        f"OpenSkill Accuracy {year}: ({(correct_predictions / total_predictions) * 100:.2f}%)",
        f"Baseline: ({(baseline_predictions / total_predictions) * 100:.2f}%)",
        f"Delta: ({((correct_predictions - baseline_predictions) / total_predictions) * 100:.2f}%)",
    )

sorted_ratings = sorted(
    ratings.items(),
    key=lambda x: sum(x[1][component].ordinal() for component in COMPONENTS),
    reverse=True,
)

for team, ratings in sorted_ratings[:10]:
    print(
        f"Team: {team}, Total Ordinal: {sum(ratings[component].ordinal() for component in COMPONENTS):.2f}"
    )
