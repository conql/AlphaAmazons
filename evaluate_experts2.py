from glob import glob
import json
import traceback
import pandas as pd
from tqdm import tqdm

bots = {}
# each bot has following statistics:
# encounter (encounter times)
# elo (elo rating)


def update_elo(winner, loser):
    k = 16

    r1 = bots[winner]["elo"] if winner in bots else 1000
    r2 = bots[loser]["elo"] if loser in bots else 1000

    r2r1 = r2 - r1
    r1r2 = r1 - r2

    if r2r1 < 4000:
        e1 = 1 / (1 + 10 ** ((r2r1) / 400))
    else:
        e1 = 0

    if r1r2 < 4000:
        e2 = 1 / (1 + 10 ** ((r1r2) / 400))
    else:
        e2 = 0

    bots[winner]["elo"] = r1 + k * (1 - e1)
    bots[loser]["elo"] = r2 + k * (0 - e2)


def process_match(match):
    if match["players"][0]["type"] != "bot" or match["players"][1]["type"] != "bot":
        return

    players = (match["players"][0]["bot"], match["players"][1]["bot"])

    for player in players:
        if player not in bots:
            bots[player] = {"encounter": 0, "elo": 1000}
        bots[player]["encounter"] += 1

    if len(match["log"]) < 2:
        return

    last_round = match["log"][-2]
    winner = None
    if "0" in last_round:
        winner = players[0]
    elif "1" in last_round:
        winner = players[1]

    if winner is None:
        return

    loser = players[0] if winner == players[1] else players[1]

    update_elo(winner, loser)


def save():
    bots_df = pd.DataFrame.from_dict(bots, orient="index")
    bots_df.to_csv("bots.csv")

if __name__ == "__main__":
    files_path = "data/raw/extracted/input/**/*.matches"
    file_list = glob(files_path, recursive=True)
    progress_bar = tqdm(total=len(file_list), unit="file")

    for i, file_path in enumerate(file_list, 1):
        # load matches
        with open(file_path, "r", encoding="utf-8") as in_f:
            matches = in_f.readlines()
            for match_str in matches:
                match = json.loads(match_str)
                try:
                    process_match(match)
                except Exception as e:
                    traceback.print_exception(type(e), e, e.__traceback__)

        progress_bar.update(1)
        
        if i % 100 == 0:
            save()
    
    save()
