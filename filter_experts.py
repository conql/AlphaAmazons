from glob import glob
from pprint import pprint
import re
import traceback
from tqdm import tqdm
import json
from check_data import md5
from game import Chessboard


def filter_files():
    with open("experts.txt", "r") as f:
        exp_ids = f.readlines()
        exp_ids = set(map(lambda x: x.strip(), exp_ids))

    output_path = "data/raw/extracted/output/expert_matches.txt"

    files_path = "data/raw/extracted/input/**/*.matches"
    file_list = glob(files_path, recursive=True)
    progress_bar = tqdm(total=len(file_list), unit="file")
    with open(output_path, "w", encoding="utf-8") as op_f:
        for file_count, file_path in enumerate(file_list, 1):
            # load matches
            with open(file_path, "r", encoding="utf-8") as in_f:
                matches = in_f.readlines()
                for match_str in matches:
                    match = json.loads(match_str)
                    try:
                        if (
                            match["players"][0]["bot"] in exp_ids
                            or match["players"][1]["bot"] in exp_ids
                        ):
                            op_f.write(match_str)
                    except Exception as e:
                        continue

            progress_bar.update(1)


def process_match(match, all_matches: set):
    chessboard = Chessboard()

    for round in match["log"]:
        player, round_value = next(iter(round.items()))
        if player != "0" and player != "1":
            continue

        player = int(player) + 1
        try:
            res = round_value["response"]
            move = [res["x0"], res["y0"], res["x1"], res["y1"], res["x2"], res["y2"]]
            chessboard.act(" ".join(map(str, move)))
        except Exception as e:
            return

        key = md5(chessboard.board)
        if key not in all_matches:
            all_matches.add(key)


def filter_match():
    all_matches = set()
    with open(
        "data/raw/extracted/output/expert_matches.txt", "r", encoding="utf-8"
    ) as file:
        count = 0
        while True:
            try:
                line = file.readline()
                if not line:
                    break

                process_match(json.loads(line), all_matches)
                count += 1
                if count % 1000 == 0:
                    print(f"count: {count}, len: {len(all_matches)}")
            except Exception as e:
                traceback.print_exception(type(e), e, e.__traceback__)


def filter_first_match():
    # load experts
    with open("experts.txt", "r", encoding="utf-8") as exp:
        expert_ids = exp.readlines()
        expert_ids = set(map(lambda x: x.strip(), expert_ids))

    pattern = r'"bot":"(.{24})"'
    with open("data/raw/extracted/output/expert_first_matches.txt", "w", encoding="utf-8") as matches:
        with open(
            "data/raw/extracted/output/expert_matches.txt", "r", encoding="utf-8"
        ) as file:
            count = 0
            while True:
                line = file.readline()
                if not line:
                    break
                count += 1
                
                ids = re.findall(pattern, line)
                if len(ids)==2 and (ids[0] in expert_ids or ids[1] in expert_ids):
                    matches.write(line)
                    expert_ids.discard(ids[0])
                    expert_ids.discard(ids[1])
                if count % 1000 == 0:
                    print(f"count: {count}, remain: {len(expert_ids)}")
                    if len(expert_ids)==0:
                        exit(0)



if __name__ == "__main__":
    filter_match()
