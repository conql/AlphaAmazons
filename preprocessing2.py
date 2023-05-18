from glob import glob
import multiprocessing
import os
import pickle
from pprint import pprint
import random
import signal
import subprocess
import time
from tqdm import tqdm
import numpy as np
from check_data import md5
import game
from game import Chessboard
import traceback


def init_pool():
    signal.signal(signal.SIGINT, signal.SIG_IGN)


def get_answer(board, player, bot_path="answer2.exe", timeout=5):
    input_str = ""

    for y in range(8):
        for x in range(8):
            t = board[y, x]
            if (t == 1 or t == 2) and player == 2:
                t = 3 - t

            input_str += str(t) + " "

        input_str += "\n"

    bot = subprocess.Popen(
        bot_path,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )

    if not bot or not bot.stdin or not bot.stdout:
        raise Exception("Failed to open bot process")

    try:
        answer, _ = bot.communicate(input=input_str, timeout=timeout)
        answer = [int(x) for x in answer.strip().split()]
    except subprocess.TimeoutExpired:
        try:
            os.kill(bot.pid, signal.SIGINT)
        except:
            pass
        raise Exception("Bot timed out after 5 seconds")

    if len(answer) != 6:
        raise Exception(f"Invalid answer: '{answer}'")

    if not game.is_valid_act(board, player, *answer):
        raise Exception(f"Invalid act: '{answer}'\n" + str(board))

    return answer


def work(args):
    try:
        (key, (init_board, _)) = args

        init_board = np.frombuffer(init_board, dtype=np.int8)
        init_board.shape = (8, 8)

        chessboard = Chessboard()
        chessboard.board = init_board.copy()
        chessboard.player = 1

        history = []

        while not chessboard.is_game_over():
            answer = get_answer(chessboard.board, chessboard.player)

            history.append((chessboard.board.copy(), chessboard.player, answer))

            chessboard.act(" ".join(map(str, answer)))

        winner = 3 - chessboard.player
        result = {}

        for board, player, answer in history:
            value = 1 if player == winner else -1
            result[md5(board.tobytes())] = (board, answer, value)

        return result
    except Exception as e:
        return e


if __name__ == "__main__":
    num_workers = 16
    input_path = "data/input/*.pickle"
    output_path = "data/output"
    block_size = 10

    # load input dict
    input_paths = glob(input_path, recursive=False)
    tasks = {}
    for path in input_paths:
        tasks = {**tasks, **pickle.load(open(path, "rb"))}

    # load output dict
    if not os.path.exists(output_path):
        os.mkdir(output_path)

    data = {}
    output_paths = glob(output_path + "/*.pickle", recursive=False)

    gross_size = 0
    for path in output_paths:
        block = pickle.load(open(path, "rb"))
        data = {**data, **block}
        gross_size += len(block)
    
    situation_count = len(data)
    print(f"Loaded {gross_size} situations, {situation_count} unique.")
    
    # filter out already processed tasks
    tasks = {k: v for k, v in tasks.items() if k not in data}

    data.clear()  # free memory


    # sort paths by name
    output_paths.sort(key=lambda x: int(os.path.basename(x).split(".")[0]))
    if len(output_paths) > 0:
        last_id = int(os.path.basename(output_paths[-1]).split(".")[0])
    else:
        last_id = 0
    current_block = {}

    # Create a pool of worker processes
    with multiprocessing.Pool(processes=num_workers, initializer=init_pool) as pool:
        try:
            # Map the work function to the worker IDs using imap_unordered
            jobs = pool.imap_unordered(work, tasks.items())
            for i, result in enumerate(tqdm(jobs, total=len(tasks), unit="match")):
                if isinstance(result, Exception):
                    traceback.print_exception(
                        type(result), result, result.__traceback__
                    )
                else:
                    for k, v in result.items():
                        current_block[k] = v
                    
                    if len(current_block) >= block_size:
                        situation_count += len(current_block)
                        pickle.dump(current_block, open(os.path.join(output_path,f"{last_id:04d}.pickle"), "wb"))
                        tqdm.write(f"Processed {situation_count} situations")
                        last_id += 1
        except KeyboardInterrupt:
            print("Caught KeyboardInterrupt, terminating workers")
            pool.terminate()
            pool.join()

            tqdm.write(f"Dumping {len(current_block)} situations. Please wait...")
            pickle.dump(current_block, open(os.path.join(output_path,f"{last_id:04d}.pickle"), "wb"))
            tqdm.write("Done. You can exit now.")
