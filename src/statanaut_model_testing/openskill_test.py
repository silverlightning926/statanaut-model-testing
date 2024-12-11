import pandas as pd
from openskill.models import PlackettLuce, PlackettLuceRating

teams = pd.read_csv("data/teams.csv")

model = PlackettLuce()

ratings = {
    team: PlackettLuceRating(name=team, mu=25.0, sigma=8.333333333333334)
    for team in teams["key"]
}

print(f"Rankings Created For {len(ratings)} Teams")

two_team_matches = pd.concat(
    [
        pd.read_csv("data/2002_matches.csv"),
        pd.read_csv("data/2003_matches.csv"),
        pd.read_csv("data/2004_matches.csv"),
    ],
    ignore_index=True,
)

three_team_matches = pd.concat(
    [
        pd.read_csv("data/2005_matches.csv"),
        pd.read_csv("data/2006_matches.csv"),
        pd.read_csv("data/2007_matches.csv"),
        pd.read_csv("data/2008_matches.csv"),
        pd.read_csv("data/2009_matches.csv"),
        pd.read_csv("data/2010_matches.csv"),
        pd.read_csv("data/2011_matches.csv"),
        pd.read_csv("data/2012_matches.csv"),
        pd.read_csv("data/2013_matches.csv"),
        pd.read_csv("data/2014_matches.csv"),
        pd.read_csv("data/2015_matches.csv"),
        pd.read_csv("data/2016_matches.csv"),
        pd.read_csv("data/2017_matches.csv"),
        pd.read_csv("data/2018_matches.csv"),
        pd.read_csv("data/2019_matches.csv"),
        pd.read_csv("data/2020_matches.csv"),
        pd.read_csv("data/2022_matches.csv"),
        pd.read_csv("data/2023_matches.csv"),
        pd.read_csv("data/2024_matches.csv"),
    ],
    ignore_index=True,
)

two_team_matches.dropna(inplace=True)
three_team_matches.dropna(inplace=True)

correct_predictions = 0

for i, match in two_team_matches.iterrows():
    red1 = ratings[match["red1"]]
    red2 = ratings[match["red2"]]
    blue1 = ratings[match["blue1"]]
    blue2 = ratings[match["blue2"]]

    red = [red1, red2]
    blue = [blue1, blue2]

    red_win = match["winning_alliance"] == "red"

    # red_win_chance, blue_win_chance = model.predict_win([red, blue])
    # if red_win_chance > blue_win_chance and red_win:
    #     correct_predictions += 1
    prediction = (red1.mu + red2.mu) > (blue1.mu + blue2.mu)
    correct_predictions += prediction == red_win

    if red_win:
        new_red, new_blue = model.rate(teams=[red, blue], ranks=[0, 1])
    else:
        new_blue, new_red = model.rate(teams=[blue, red], ranks=[0, 1])

    ratings[match["red1"]] = new_red[0]
    ratings[match["red2"]] = new_red[1]
    ratings[match["blue1"]] = new_blue[0]
    ratings[match["blue2"]] = new_blue[1]

    print(f"Match {match["event_key"]} - {i} Updated")

for i, match in three_team_matches.iterrows():
    red1 = ratings[match["red1"]]
    red2 = ratings[match["red2"]]
    red3 = ratings[match["red3"]]
    blue1 = ratings[match["blue1"]]
    blue2 = ratings[match["blue2"]]
    blue3 = ratings[match["blue3"]]

    red = [red1, red2, red3]
    blue = [blue1, blue2, blue3]

    red_win = match["winning_alliance"] == "red"

    # red_win_chance, blue_win_chance = model.predict_win([red, blue])
    # if red_win_chance > blue_win_chance and red_win:
    #     correct_predictions += 1
    prediction = (red1.mu + red2.mu + red3.mu) > (blue1.mu + blue2.mu + blue3.mu)
    correct_predictions += prediction == red_win

    if red_win:
        new_red, new_blue = model.rate(teams=[red, blue], ranks=[0, 1])
    else:
        new_blue, new_red = model.rate(teams=[blue, red], ranks=[0, 1])

    ratings[match["red1"]] = new_red[0]
    ratings[match["red2"]] = new_red[1]
    ratings[match["red3"]] = new_red[2]
    ratings[match["blue1"]] = new_blue[0]
    ratings[match["blue2"]] = new_blue[1]
    ratings[match["blue3"]] = new_blue[2]

    print(f"Match {match["event_key"]} - {i} Updated")

print("Ratings Updated")


sorted_ratings = sorted(ratings.items(), key=lambda x: x[1].mu, reverse=True)
top_teams = sorted_ratings[:5]

print("Top 5 Teams:")
for team, rating in top_teams:
    print(f"Team {team}: Rating {rating}")

print(
    f"Correct Predictions: {(correct_predictions / (len(two_team_matches) + len(three_team_matches)))*100}%"
)
