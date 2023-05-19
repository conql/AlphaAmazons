from glob import glob
import json
import signal
import traceback
from tqdm import tqdm
import pandas as pd

from game import Chessboard
from preprocessing2 import get_answer
from multiprocessing import Process, Manager, Pool

bots = {}
# each bot has following statistics:
# evaluate (evaluate times)
# correct (correct times)
# encounter (encounter times)


def evaluate_bot(match, bot_id):
    if match["players"][0]["bot"] == bot_id:
        role = 1
    elif match["players"][1]["bot"] == bot_id:
        role = 2
    else:
        raise Exception("Bot id not found in match")

    chessboard = Chessboard()
    eval_count = 0
    correct_count = 0

    # for round in tqdm(match["log"], desc=f"Evaluating {bot_id}", unit="round"):
    for round in match["log"]:
        player, round_value = next(iter(round.items()))
        if player != "0" and player != "1":
            continue

        player = int(player) + 1

        res = round_value["response"]
        move = [res["x0"], res["y0"], res["x1"], res["y1"], res["x2"], res["y2"]]

        if role == player:
            eval_count += 1
            ans = get_answer(chessboard.board, player)
            if move == ans:
                correct_count += 1
        chessboard.act(" ".join(map(str, move)))

    return (correct_count, eval_count)


def process_match(arg):
    match, bots = arg
    players = match["players"]

    for player in players:
        if player["type"] == "bot":
            bot_id = player["bot"]

            if bot_id not in bots:
                bots[bot_id] = {"evaluate": 0, "correct": 0, "encounter": 1}
            else:
                bots[bot_id] = {
                    "evaluate": bots[bot_id]["evaluate"],
                    "correct": bots[bot_id]["correct"],
                    "encounter": bots[bot_id]["encounter"] + 1,
                }

            try:
                (correct, eval) = evaluate_bot(match, bot_id)
                # bots[bot_id]["evaluate"] += eval
                # bots[bot_id]["correct"] += correct

                bots[bot_id] = {
                    "evaluate": bots[bot_id]["evaluate"] + eval,
                    "correct": bots[bot_id]["correct"] + correct,
                    "encounter": bots[bot_id]["encounter"],
                }
            except Exception as e:
                continue


def init_pool():
    signal.signal(signal.SIGINT, signal.SIG_IGN)


def save_data(bots):
    # convert bots to csv and save
    bots_df = pd.DataFrame.from_dict(bots, orient="index")
    bots_df.to_csv("bots.csv")


if __name__ == "__main__":
    manager = Manager()
    bots = manager.dict()

    with Pool(processes=16, initializer=init_pool) as pool:
        try:
            with open("data/raw/extracted/output/expert_first_matches.txt", "r") as f:
                matches = f.readlines()
                args = [(json.loads(match), bots) for match in matches]
                jobs = pool.imap_unordered(process_match, args)

                for count, result in enumerate(tqdm(jobs, total=len(matches)), 1):
                    if isinstance(result, Exception):
                        tqdm.write(f"Error in {count}:")
                        traceback.print_exception(
                            type(result), result, result.__traceback__
                        )

                    save_data(bots)
        except KeyboardInterrupt:
            print("KeyboardInterrupt, terminating...")
            pool.terminate()
            pool.join()

            tqdm.write("Please wait for the processes to terminate...")
            save_data(bots)
