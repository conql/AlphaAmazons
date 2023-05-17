import copy
from datetime import datetime
import hashlib
import json
import multiprocessing
import os
from pprint import pprint
import random
import signal
import subprocess
import lmdb
import torch
from tqdm import tqdm
import glob
import neptune


class Database:
    def __init__(self, path="data/lmdb"):
        self.env = lmdb.open(path=path, map_size=2**29)

    def get(self, key):
        with self.env.begin() as txn:
            return txn.get(key.encode())

    def put(self, key, value):
        with self.env.begin(write=True) as txn:
            txn.put(key.encode(), value.encode())


class Chessboard:
    def __init__(self):
        self.history = []
        self.board = [[0 for _ in range(8)] for _ in range(8)]
        self.init_board()
        self.current_player = 1

    def init_board(self):
        black_positions = [(0, 2), (2, 0), (5, 0), (7, 2)]
        white_positions = [(0, 5), (2, 7), (5, 7), (7, 5)]

        for position in black_positions:
            x, y = position
            self.board[y][x] = 1

        for position in white_positions:
            x, y = position
            self.board[y][x] = 2

    def is_valid_move(self, x0, y0, x1, y1):
        dx, dy = abs(x1 - x0), abs(y1 - y0)
        if self.board[y0][x0] != self.current_player or self.board[y1][x1] != 0:
            return False
        if dx == dy or dx == 0 or dy == 0:
            step_x = (x1 - x0) // max(1, abs(x1 - x0))
            step_y = (y1 - y0) // max(1, abs(y1 - y0))
            x, y = x0 + step_x, y0 + step_y
            while x != x1 or y != y1:
                if self.board[y][x] != 0:
                    return False
                x += step_x
                y += step_y
            return True
        return False

    def is_valid_obstacle(self, x1, y1, x2, y2):
        dx, dy = abs(x2 - x1), abs(y2 - y1)
        if self.board[y2][x2] != 0:
            return False
        if dx == dy or dx == 0 or dy == 0:
            step_x = (x2 - x1) // max(1, abs(x2 - x1))
            step_y = (y2 - y1) // max(1, abs(y2 - y1))
            x, y = x1 + step_x, y1 + step_y
            while x != x2 or y != y2:
                if self.board[y][x] != 0:
                    return False
                x += step_x
                y += step_y
            return True
        return False

    def add(self, response):
        x0, y0, x1, y1, x2, y2 = map(int, response.split())

        if not self.is_valid_move(x0, y0, x1, y1):
            # pprint(self.getData())
            raise ValueError("Invalid move.")

        # Move the piece
        self.board[y0][x0] = 0
        self.board[y1][x1] = self.current_player

        if not self.is_valid_obstacle(x1, y1, x2, y2):
            self.board[y0][x0] = self.current_player
            self.board[y1][x1] = 0

            # pprint(self.getData())
            raise ValueError("Invalid obstacle placement.")

        # Add the obstacle
        self.board[y2][x2] = 3

        # Add the move to history
        self.history.append(response)

        # Switch the current player
        self.current_player = 3 - self.current_player

    def revert(self):
        assert (
            torch.sum(torch.tensor(self.board) == 1) == 4
            and torch.sum(torch.tensor(self.board) == 2) == 4
        )

        response = self.history[-1]
        # Remove the move from history
        self.history.pop()

        # Switch the current player
        self.current_player = 3 - self.current_player

        x0, y0, x1, y1, x2, y2 = map(int, response.split())

        assert self.board[y1][x1] == self.current_player

        # Remove the obstacle
        self.board[y2][x2] = 0

        # Move back the piece
        self.board[y0][x0] = self.current_player
        self.board[y1][x1] = 0

        assert (
            torch.sum(torch.tensor(self.board) == 1) == 4
            and torch.sum(torch.tensor(self.board) == 2) == 4
        )

    def getData(self):
        return self.board

    # convert data to multi line string, with the delimiter being space and newline
    def getDataStr(self):
        data_str = ""
        for row in self.board:
            row_str = " ".join(str(cell) for cell in row)
            data_str += row_str + "\n"  # type: ignore
        return data_str.strip()

    def hash_board(self):
        board_str = self.getDataStr()
        md5 = hashlib.md5()
        md5.update(board_str.encode())
        board_hash = md5.hexdigest()
        return board_hash

    def dump(self, database: Database):
        id = self.hash_board()
        stored = database.get(id)

        if stored:
            # print(f"Board {id} already exists.")
            return

        # Get answer
        answer = self.getAnswer()
        data = [self.getData(), answer]

        # Store the board
        database.put(id, json.dumps(data))

    def getAnswer(self, bot_path="answer.exe"):
        current_bot = subprocess.Popen(
            bot_path,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )

        if not current_bot.stdin or not current_bot.stdout:
            raise ValueError("Invalid bot path.")

        history = self.history.copy()
        if self.current_player == 1:
            history = ["-1 -1 -1 -1 -1 -1"] + history

        try:
            history = f"{(len(history) // 2)+1}\n" + "\n".join(history) + "\n"
            answer, _ = current_bot.communicate(input=history, timeout=5)
            answer = answer.strip()
        except subprocess.TimeoutExpired:
            try:
                os.kill(
                    current_bot.pid, signal.SIGINT
                )  # Send a signal to terminate the process
            except:
                pass
            raise RuntimeError("Bot timed out after 2 seconds.")

        try:
            original = copy.deepcopy(self.board)
            self.add(answer)
            self.revert()
            assert original == self.board

        except Exception as e:
            raise RuntimeError(f"Bot returned an invalid answer: {e}")

        return answer


def process_match(line):
    # global total_matches
    # global total_errors

    # with total_matches.get_lock():
    #     total_matches.value += 1  # type: ignore
    try:
        game_data = json.loads(line)["log"]  # array
        game_data.pop(0)
        game_data = game_data[::2]

        game_data = [entry[list(entry.keys())[0]]["response"] for entry in game_data]
        game_data = [
            f"{round['x0']} {round['y0']} {round['x1']} {round['y1']} {round['x2']} {round['y2']}"
            for round in game_data
        ]

        chessboard = Chessboard()
        database = Database()

        for i, round in enumerate(game_data):
            chessboard.dump(database)
            chessboard.add(round)
    except Exception as e:
        # with total_errors.get_lock():
        #     total_errors.value += 1  # type: ignore

        if isinstance(e, KeyboardInterrupt):
            raise e
        else:
            with open("error.log", "a") as log_file:
                # Get timestamp
                timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                print(f"[{timestamp}] Error processing match: {e}", file=log_file)


if __name__ == "__main__":
    files_path = "data/raw/extracted/**/*.matches"
    file_list = glob.glob(files_path, recursive=True)

    # run = neptune.init_run(project="conql/Amazons-preprocess")

    random.shuffle(file_list)

    file_progress = tqdm(file_list, desc="Processing files", unit="file")
    for file_path in file_progress:
        file_progress.set_description(f"Processing file: {file_path}")

        with open(file_path, "r", encoding="utf-8") as f:
            matches = f.readlines()

            # total_matches = multiprocessing.Value("i", 0)
            # total_errors = multiprocessing.Value("i", 0)

            # for match in tqdm(
            #     matches, desc="Processing matches", unit="match", leave=False
            # ):
            #     process_match(match)

            original_sigint_handler = signal.signal(signal.SIGINT, signal.SIG_IGN)
            with multiprocessing.Pool(10) as pool:
                signal.signal(signal.SIGINT, original_sigint_handler)
                try:
                    for p in tqdm(
                        pool.imap(process_match, matches),
                        desc="Processing matches",
                        unit="match",
                        total=len(matches),
                        leave=False,
                    ):
                        # run["total_matches"] = total_matches.value  # type: ignore
                        # run["total_errors"] = total_errors.value  # type: ignore
                        pass

                except KeyboardInterrupt:
                    print("Caught KeyboardInterrupt, terminating workers")
                    pool.terminate()
                    pool.join()
                    raise
