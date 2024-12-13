from pprint import pprint
import pandas as pd
from math import isclose
from openskill.models import (
    PlackettLuce,
    PlackettLuceRating,
    BradleyTerryFullRating,
    BradleyTerryFull,
    ThurstoneMostellerFull,
    ThurstoneMostellerFullRating,
    ThurstoneMostellerPart,
    ThurstoneMostellerPartRating,
    BradleyTerryPart,
    BradleyTerryPartRating,
)
from json import loads

AUTO_WEIGHT = 0.3
TELEOP_WEIGHT = 0.4
ENDGAME_WEIGHT = 0.2
FOUL_WEIGHT = 0.1

assert isclose(AUTO_WEIGHT + TELEOP_WEIGHT + ENDGAME_WEIGHT + FOUL_WEIGHT, 1.0)

teams = pd.read_csv("data/teams.csv")

model = PlackettLuce()

ratings = {
    team: [
        PlackettLuceRating(name=team, mu=25.0, sigma=8.333333333333334),
        PlackettLuceRating(name=team, mu=25.0, sigma=8.333333333333334),
        PlackettLuceRating(name=team, mu=25.0, sigma=8.333333333333334),
        PlackettLuceRating(name=team, mu=25.0, sigma=8.333333333333334),
    ]
    for team in teams["key"]
}

print(f"Rankings Created For {len(ratings)} Teams")

years = list(range(2016, 2025))

accuracy_by_year = {}

for year in years:

    try:
        year_matches = f"data/{year}_matches.csv"
        matches = pd.read_csv(year_matches)
    except FileNotFoundError:
        print(f"No match file found for year {year}, skipping.")
        continue

    matches["red_score"] = matches["red_score"].astype(float)
    matches["blue_score"] = matches["blue_score"].astype(float)
    matches.dropna(
        subset=[
            "red1",
            "red2",
            "red3",
            "red_score_breakdown",
            "blue1",
            "blue2",
            "blue3",
            "blue_score_breakdown",
        ],
        inplace=True,
    )

    correct_predictions = 0
    total_predictions = 0

    for i, match in matches.iterrows():
        red1 = ratings[match["red1"]]
        red2 = ratings[match["red2"]]
        red3 = ratings[match["red3"]]
        blue1 = ratings[match["blue1"]]
        blue2 = ratings[match["blue2"]]
        blue3 = ratings[match["blue3"]]

        red_team_auto = [red1[0], red2[0], red3[0]]
        blue_team_auto = [blue1[0], blue2[0], blue3[0]]

        red_team_teleop = [red1[1], red2[1], red3[1]]
        blue_team_teleop = [blue1[1], blue2[1], blue3[1]]

        red_team_endgame = [red1[2], red2[2], red3[2]]
        blue_team_endgame = [blue1[2], blue2[2], blue3[2]]

        red_team_foul = [red1[3], red2[3], red3[3]]
        blue_team_foul = [blue1[3], blue2[3], blue3[3]]

        red_score = match["red_score"]
        blue_score = match["blue_score"]

        red_score_breakdown = loads(match["red_score_breakdown"])
        blue_score_breakdown = loads(match["blue_score_breakdown"])

        red_auto = red_score_breakdown["autoPoints"]
        # red_teleop = red_score_breakdown["teleopPoints"]

        blue_auto = blue_score_breakdown["autoPoints"]
        # blue_teleop = blue_score_breakdown["teleopPoints"]

        if year == 2016:
            red_teleop = (
                red_score_breakdown["teleopCrossingPoints"]
                + red_score_breakdown["teleopBoulderPoints"]
            )
            blue_teleop = (
                blue_score_breakdown["teleopCrossingPoints"]
                + blue_score_breakdown["teleopBoulderPoints"]
            )

            red_endgame = red_score_breakdown["teleopPoints"] - red_teleop
            blue_endgame = blue_score_breakdown["teleopPoints"] - blue_teleop
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
                red_score_breakdown["cargoPoints"]
                + red_score_breakdown["hatchPanelPoints"]
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

        red_foul = blue_score_breakdown["foulPoints"]
        blue_foul = red_score_breakdown["foulPoints"]

        winner = (
            "red"
            if red_score > blue_score
            else "blue" if blue_score > red_score else None
        )

        prediction = (
            (
                (red_team_auto[0].mu + red_team_auto[1].mu + red_team_auto[2].mu)
                * AUTO_WEIGHT
            )
            + (
                (red_team_teleop[0].mu + red_team_teleop[1].mu + red_team_teleop[2].mu)
                * TELEOP_WEIGHT
            )
            + (
                (
                    red_team_endgame[0].mu
                    + red_team_endgame[1].mu
                    + red_team_endgame[2].mu
                )
                * ENDGAME_WEIGHT
            )
            - (
                (red_team_foul[0].mu + red_team_foul[1].mu + red_team_foul[2].mu)
                * FOUL_WEIGHT
            )
        ) > (
            (
                (blue_team_auto[0].mu + blue_team_auto[1].mu + blue_team_auto[2].mu)
                * AUTO_WEIGHT
            )
            + (
                (
                    blue_team_teleop[0].mu
                    + blue_team_teleop[1].mu
                    + blue_team_teleop[2].mu
                )
                * TELEOP_WEIGHT
            )
            + (
                (
                    blue_team_endgame[0].mu
                    + blue_team_endgame[1].mu
                    + blue_team_endgame[2].mu
                )
                * ENDGAME_WEIGHT
            )
            - (
                (blue_team_foul[0].mu + blue_team_foul[1].mu + blue_team_foul[2].mu)
                * FOUL_WEIGHT
            )
        )
        correct_predictions += prediction == (winner == "red")
        total_predictions += 1

        # if winner == "red":
        #     new_auto_red, new_auto_blue = auto_model.rate(
        #         teams=[red_team_auto, blue_team_auto],
        #         ranks=[0, 1],
        #     )
        #     new_teleop_red, new_teleop_blue = teleop_model.rate(
        #         teams=[red_team_teleop, blue_team_teleop],
        #         ranks=[0, 1],
        #     )
        #     new_endgame_red, new_endgame_blue = end_game_model.rate(
        #         teams=[red_team_endgame, blue_team_endgame],
        #         ranks=[0, 1],
        #     )
        #     new_foul_red, new_foul_blue = foul_model.rate(
        #         teams=[red_team_foul, blue_team_foul],
        #         ranks=[0, 1],
        #     )

        # elif winner == "blue":
        #     new_auto_blue, new_auto_red = auto_model.rate(
        #         teams=[blue_team_auto, red_team_auto],
        #         ranks=[0, 1],
        #     )
        #     new_teleop_blue, new_teleop_red = teleop_model.rate(
        #         teams=[blue_team_teleop, red_team_teleop],
        #         ranks=[0, 1],
        #     )
        #     new_endgame_blue, new_endgame_red = end_game_model.rate(
        #         teams=[blue_team_endgame, red_team_endgame],
        #         ranks=[0, 1],
        #     )
        #     new_foul_blue, new_foul_red = foul_model.rate(
        #         teams=[blue_team_foul, red_team_foul],
        #         ranks=[0, 1],
        #     )

        # else:
        #     new_auto_red, new_auto_blue = auto_model.rate(
        #         teams=[red_team_auto, blue_team_auto],
        #         ranks=[0, 0],
        #     )
        #     new_teleop_red, new_teleop_blue = teleop_model.rate(
        #         teams=[red_team_teleop, blue_team_teleop],
        #         ranks=[0, 0],
        #     )
        #     new_endgame_red, new_endgame_blue = end_game_model.rate(
        #         teams=[red_team_endgame, blue_team_endgame],
        #         ranks=[0, 0],
        #     )
        #     new_foul_red, new_foul_blue = foul_model.rate(
        #         teams=[red_team_foul, blue_team_foul],
        #         ranks=[0, 0],
        #     )

        if red_auto > blue_auto:
            new_auto_red, new_auto_blue = model.rate(
                teams=[red_team_auto, blue_team_auto],
                ranks=[0, 1],
            )

        elif blue_auto > red_auto:
            new_auto_blue, new_auto_red = model.rate(
                teams=[blue_team_auto, red_team_auto],
                ranks=[0, 1],
            )

        else:
            new_auto_red, new_auto_blue = model.rate(
                teams=[red_team_auto, blue_team_auto],
                ranks=[0.5, 0.5],
            )

        if red_teleop > blue_teleop:
            new_teleop_red, new_teleop_blue = model.rate(
                teams=[red_team_teleop, blue_team_teleop],
                ranks=[0, 1],
            )

        elif blue_teleop > red_teleop:
            new_teleop_blue, new_teleop_red = model.rate(
                teams=[blue_team_teleop, red_team_teleop],
                ranks=[0, 1],
            )

        else:
            new_teleop_red, new_teleop_blue = model.rate(
                teams=[red_team_teleop, blue_team_teleop],
                ranks=[0.5, 0.5],
            )

        if red_endgame > blue_endgame:
            new_endgame_red, new_endgame_blue = model.rate(
                teams=[red_team_endgame, blue_team_endgame],
                ranks=[0, 1],
            )

        elif blue_endgame > red_endgame:
            new_endgame_blue, new_endgame_red = model.rate(
                teams=[blue_team_endgame, red_team_endgame],
                ranks=[0, 1],
            )

        else:
            new_endgame_red, new_endgame_blue = model.rate(
                teams=[red_team_endgame, blue_team_endgame],
                ranks=[0.5, 0.5],
            )

        if red_foul > blue_foul:
            new_foul_red, new_foul_blue = model.rate(
                teams=[red_team_foul, blue_team_foul],
                ranks=[0, 1],
            )

        elif blue_foul > red_foul:
            new_foul_blue, new_foul_red = model.rate(
                teams=[blue_team_foul, red_team_foul],
                ranks=[0, 1],
            )

        else:
            new_foul_red, new_foul_blue = model.rate(
                teams=[red_team_foul, blue_team_foul],
                ranks=[0.5, 0.5],
            )

        for i, key in enumerate(["red1", "red2", "red3"]):
            ratings[match[key]][0] = new_auto_red[i]
            ratings[match[key]][1] = new_teleop_red[i]
            ratings[match[key]][2] = new_endgame_red[i]
            ratings[match[key]][3] = new_foul_red[i]

        for i, key in enumerate(["blue1", "blue2", "blue3"]):
            ratings[match[key]][0] = new_auto_blue[i]
            ratings[match[key]][1] = new_teleop_blue[i]
            ratings[match[key]][2] = new_endgame_blue[i]
            ratings[match[key]][3] = new_foul_blue[i]

    if total_predictions > 0:
        accuracy = (correct_predictions / total_predictions) * 100
        accuracy_by_year[year] = accuracy
        print(f"Year {year} Prediction Accuracy: {accuracy:.2f}%")
    else:
        print(f"No predictions for year {year}")

sorted_ratings = sorted(
    ratings.items(),
    key=lambda x: x[1][0].mu * AUTO_WEIGHT
    + x[1][1].mu * TELEOP_WEIGHT
    + x[1][2].mu * ENDGAME_WEIGHT
    - x[1][3].mu * FOUL_WEIGHT,
    reverse=True,
)

print("Top 5 Teams:")
for team, rating in sorted_ratings[:5]:
    print(
        team,
        f"""Total: {rating[0].mu * AUTO_WEIGHT
        + rating[1].mu * TELEOP_WEIGHT
        + rating[2].mu * ENDGAME_WEIGHT
        - rating[3].mu * FOUL_WEIGHT} - Auto: {rating[0].mu}, Tele-op: {rating[1].mu}, "Endgame": {rating[2].mu}, "Foul": {rating[3].mu}""",
    )
