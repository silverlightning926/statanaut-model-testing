import pandas as pd
from json import loads
import matplotlib.pyplot as plt
from openskill.models import BradleyTerryFull, BradleyTerryFullRating

PRINT_TEAMS = False
NUM_TEAMS = 20
SHOW_PLOT = False

STARTING_MU = 25.0
STARTING_SIGMA = STARTING_MU / 3.0

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

COMPONENTS = ["auto", "teleop", "endgame"]

ratings = {}

accuracy_over_time = []

model = BradleyTerryFull()


def print_top_teams(ratings: dict, year: int, num_teams: int = NUM_TEAMS) -> None:
    if not PRINT_TEAMS:
        return

    if year < 2016:
        sorted_ratings = sorted(
            ratings.items(),
            key=lambda x: x[1].mu,
            reverse=True,
        )

        for team, rating in sorted_ratings[:num_teams]:
            print(f"{team} - {rating.mu:.2f}")

    else:
        sorted_ratings = sorted(
            ratings.items(),
            key=lambda x: sum([x[1][component].mu for component in COMPONENTS]),
            reverse=True,
        )

        for team, rating in sorted_ratings[:num_teams]:
            print(
                f"{team} - {sum([rating[component].mu for component in COMPONENTS]):.2f} | auto: {rating['auto'].mu:.2f} | teleop: {rating['teleop'].mu:.2f} | endgame: {rating['endgame'].mu:.2f}"
            )


def get_events(matches_df: pd.DataFrame) -> list:
    return matches_df["event_key"].drop_duplicates().tolist()


def get_event_teams(matches_df: pd.DataFrame, event: str) -> list:
    return list(
        filter(
            lambda x: pd.notna(x),
            set(matches_df[matches_df["event_key"] == event]["red1"])
            | set(matches_df[matches_df["event_key"] == event]["red2"])
            | set(matches_df[matches_df["event_key"] == event]["red3"])
            | set(matches_df[matches_df["event_key"] == event]["blue1"])
            | set(matches_df[matches_df["event_key"] == event]["blue2"])
            | set(matches_df[matches_df["event_key"] == event]["blue3"]),
        )
    )


def filter_matches(matches_df: pd.DataFrame, year: int) -> pd.DataFrame:
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

    if year < 2005:
        matches_df = matches_df.dropna(subset=["red1", "red2", "blue1", "blue2"])
    else:
        matches_df = matches_df.dropna(
            subset=["red1", "red2", "red3", "blue1", "blue2", "blue3"]
        )

    return matches_df


def print_stats(
    year: int,
    correct_predictions: int,
    baseline_predictions: int,
    total_predictions: int,
) -> None:
    print(
        f"{year} - ✅ ({(correct_predictions / total_predictions) * 100:.2f}%) |",
        f"⬜ ({(baseline_predictions / total_predictions) * 100:.2f}%) |",
        f"🔼 ({((correct_predictions - baseline_predictions) / total_predictions) * 100:.2f}%)",
    )


for year in range(2002, 2016):
    try:
        matches_df = pd.read_csv(f"../../data/matches/{year}_matches.csv")
    except FileNotFoundError:
        continue

    matches_df = filter_matches(matches_df, year)

    events = get_events(matches_df)

    baseline_predictions = 0
    correct_predictions = 0
    total_predictions = 0

    for event in events:
        event_matches = matches_df[matches_df["event_key"] == event]

        event_teams = get_event_teams(matches_df, event)

        for teams in event_teams:
            if teams not in ratings:
                ratings[teams] = BradleyTerryFullRating(
                    name=teams, mu=STARTING_MU, sigma=STARTING_SIGMA
                )

            else:
                ratings[teams].sigma = STARTING_SIGMA

        for _, match in event_matches.iterrows():
            red_alliance = [match["red1"], match["red2"]]
            if pd.notna(match["red3"]):
                red_alliance.append(match["red3"])

            blue_alliance = [match["blue1"], match["blue2"]]
            if pd.notna(match["blue3"]):
                blue_alliance.append(match["blue3"])

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

            total_predictions += 1
            if winner == "red":
                baseline_predictions += 1
            if red_pred > blue_pred and winner == "red":
                correct_predictions += 1
            elif red_pred < blue_pred and winner == "blue":
                correct_predictions += 1
            elif red_pred == blue_pred and winner == "tie":
                correct_predictions += 1

            new_red_ratings, new_blue_ratings = model.rate(
                teams=[red_ratings, blue_ratings],
                scores=[red_score, blue_score],
            )

            for i, team in enumerate(red_alliance):
                ratings[team] = new_red_ratings[i]

            for i, team in enumerate(blue_alliance):
                ratings[team] = new_blue_ratings[i]

    accuracy_over_time.append(
        (
            year,
            ((correct_predictions / total_predictions) * 100),
            ((baseline_predictions / total_predictions) * 100),
            (
                (correct_predictions / total_predictions)
                - (baseline_predictions / total_predictions)
            )
            * 100,
        )
    )

    print_stats(year, correct_predictions, baseline_predictions, total_predictions)
    print_top_teams(ratings, year)

for team in ratings:
    legacy_rating = ratings[team]
    ratings[team] = {
        "auto": legacy_rating,
        "teleop": legacy_rating,
        "endgame": legacy_rating,
    }


def breakdown_match(match, year) -> tuple:

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

    point_keys = teleop_mapping[year]

    red_teleop = sum(red_score_breakdown[key] for key in point_keys)
    blue_teleop = sum(blue_score_breakdown[key] for key in point_keys)

    if year == 2023:
        red_endgame = (
            red_score_breakdown["teleopPoints"] + red_score_breakdown["linkPoints"]
        ) - red_teleop
        blue_endgame = (
            blue_score_breakdown["teleopPoints"] + blue_score_breakdown["linkPoints"]
        ) - blue_teleop

    else:
        red_endgame = red_score_breakdown["teleopPoints"] - red_teleop
        blue_endgame = blue_score_breakdown["teleopPoints"] - blue_teleop

    return (
        red_auto,
        red_teleop,
        red_endgame,
        blue_auto,
        blue_teleop,
        blue_endgame,
    )


for year in range(2016, 2025):
    try:
        matches_df = pd.read_csv(f"../../data/matches/{year}_matches.csv")
    except FileNotFoundError:
        continue

    matches_df = filter_matches(matches_df, year)

    matches_df["red_score_breakdown"] = matches_df["red_score_breakdown"].apply(loads)
    matches_df["blue_score_breakdown"] = matches_df["blue_score_breakdown"].apply(loads)

    events = get_events(matches_df)

    baseline_predictions = 0
    correct_predictions = 0
    total_predictions = 0

    for event in events:
        event_matches = matches_df[matches_df["event_key"] == event]

        event_teams = get_event_teams(matches_df, event)

        for teams in event_teams:
            if teams not in ratings:
                ratings[teams] = {
                    component: BradleyTerryFullRating(
                        name=f"{teams}_{component}",
                        mu=STARTING_MU,
                        sigma=STARTING_SIGMA,
                    )
                    for component in COMPONENTS
                }

            else:
                for component in COMPONENTS:
                    ratings[teams][component].sigma = STARTING_SIGMA

        for _, match in event_matches.iterrows():
            red_alliance = [match["red1"], match["red2"], match["red3"]]
            blue_alliance = [match["blue1"], match["blue2"], match["blue3"]]

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

            outcome = (
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

            total_predictions += 1
            if outcome == "red":
                baseline_predictions += 1
            if red_pred > blue_pred and outcome == "red":
                correct_predictions += 1
            elif red_pred < blue_pred and outcome == "blue":
                correct_predictions += 1
            elif red_pred == blue_pred and outcome == "tie":
                correct_predictions += 1

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

                red_component_ratings = [
                    ratings[team][component] for team in red_alliance
                ]
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

    accuracy_over_time.append(
        (
            year,
            ((correct_predictions / total_predictions) * 100),
            ((baseline_predictions / total_predictions) * 100),
            (
                (correct_predictions / total_predictions)
                - (baseline_predictions / total_predictions)
            )
            * 100,
        )
    )

    print_stats(year, correct_predictions, baseline_predictions, total_predictions)
    print_top_teams(ratings, year)


if SHOW_PLOT:
    years = [year for year, _, _, _ in accuracy_over_time]
    accuracy = [accuracy for _, accuracy, _, _ in accuracy_over_time]
    baseline = [baseline for _, _, baseline, _ in accuracy_over_time]
    delta = [delta for _, _, _, delta in accuracy_over_time]

    plt.figure(figsize=(16, 8))

    plt.plot(years, accuracy, label="Model Accuracy", marker="o")
    plt.plot(years, baseline, label="Baseline Accuracy", marker="o")
    plt.plot(years, delta, label="Delta", marker="o")

    plt.xticks(sorted(set(year for year, _, _, _ in accuracy_over_time)))

    plt.xlabel("Year")
    plt.ylabel("Accuracy (%)")

    plt.title("Model Accuracy Over Time")

    plt.legend()

    plt.show()