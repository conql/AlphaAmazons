from glob import glob
import multiprocessing
import os
import pickle
import signal
import subprocess
import traceback
import numpy as np

from tqdm import tqdm
import game


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
        (key, (board, old_answer)) = args

        new_answer = get_answer(board, board[old_answer[1], old_answer[0]])
        new_answer = np.array(new_answer, dtype=np.int8)

        return key, (board, new_answer)

    except Exception as e:
        return e


if __name__ == "__main__":
    num_workers = 16
    input_path = "data/verify/*.pickle"
    output_path = "data/verify2"
    block_size = 10000

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

    sc = len(data)
    print(f"Loaded {gross_size} situations, {sc} unique.")

    # filter out already processed tasks
    tasks = {k: v for k, v in tasks.items() if k not in data}

    # sort paths by name
    output_paths.sort(key=lambda x: int(os.path.basename(x).split(".")[0]))
    if len(output_paths) > 0:
        last_id = int(os.path.basename(output_paths[-1]).split(".")[0])
    else:
        last_id = 0

    current_block = {}
    wrong = 0

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
                    k, v = result
                    if list(tasks[k][1]) != list(v[1]):
                        wrong += 1

                    current_block[k] = v

                    if i % 50 == 0 or len(current_block) >= block_size:
                        pickle.dump(
                            current_block,
                            open(
                                os.path.join(output_path, f"{last_id:04d}.pickle"), "wb"
                            ),
                        )
                        tqdm.write(
                            f"Processed {i+1} situations. {wrong} is wrong. ({wrong/(i+1)*100:.02f}%)"
                        )

                        if len(current_block) >= block_size:
                            last_id += 1
                            current_block = {}

        except KeyboardInterrupt:
            print("Caught KeyboardInterrupt, terminating workers")
            pool.terminate()
            pool.join()

            tqdm.write(f"Dumping {len(current_block)} situations. Please wait...")
            pickle.dump(
                current_block,
                open(os.path.join(output_path, f"{last_id:04d}.pickle"), "wb"),
            )
            tqdm.write("Done. You can exit now.")
