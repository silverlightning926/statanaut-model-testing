import pandas as pd
from openskill.models import PlackettLuce, PlackettLuceRating

teams = pd.read_csv("data/teams.csv")

model = PlackettLuce()

ratings = {
    team: PlackettLuceRating(name=team, mu=25.0, sigma=8.333333333333334)
    for team in teams["key"]
}

print(f"Rankings Created For {len(ratings)} Teams")

years = list(range(2002, 2025))

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
    matches.dropna(subset=["red1", "red2", "blue1", "blue2"], inplace=True)

    correct_predictions = 0
    total_predictions = 0

    for i, match in matches.iterrows():
        red1 = ratings[match["red1"]]
        red2 = ratings[match["red2"]]
        red3 = ratings[match["red3"]] if pd.notna(match["red3"]) else None
        blue1 = ratings[match["blue1"]]
        blue2 = ratings[match["blue2"]]
        blue3 = ratings[match["blue3"]] if pd.notna(match["blue3"]) else None

        red = [red1, red2]
        blue = [blue1, blue2]
        if red3:
            red.append(red3)
        if blue3:
            blue.append(blue3)

        red_score = match["red_score"]
        blue_score = match["blue_score"]

        winner = (
            "red"
            if red_score > blue_score
            else "blue" if blue_score > red_score else None
        )

        if sum(player.mu for player in red) > sum(player.mu for player in blue):
            correct_predictions += winner == "red"
        else:
            correct_predictions += winner == "blue"
        total_predictions += 1

        if winner == "red":
            new_red, new_blue = model.rate(
                teams=[red, blue],
                ranks=[0, 1],
            )
        elif winner == "blue":
            new_blue, new_red = model.rate(
                teams=[blue, red],
                ranks=[0, 1],
            )
        else:
            new_red, new_blue = model.rate(
                teams=[red, blue],
                ranks=[1, 1],
            )

        ratings[match["red1"]] = new_red[0]
        ratings[match["red2"]] = new_red[1]
        if red3:
            ratings[match["red3"]] = new_red[2]
        ratings[match["blue1"]] = new_blue[0]
        ratings[match["blue2"]] = new_blue[1]
        if blue3:
            ratings[match["blue3"]] = new_blue[2]

    if total_predictions > 0:
        accuracy = (correct_predictions / total_predictions) * 100
        accuracy_by_year[year] = accuracy
        print(f"Year {year} Prediction Accuracy: {accuracy:.2f}%")
    else:
        print(f"No predictions for year {year}")
