import hashlib
import os
import subprocess
from pprint import pprint


class Chessboard:
    def __init__(self):
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
            pprint(self.getData())
            raise ValueError("Invalid move.")

        # Move the piece
        self.board[y0][x0] = 0
        self.board[y1][x1] = self.current_player

        if not self.is_valid_obstacle(x1, y1, x2, y2):
            pprint(self.getData())
            raise ValueError("Invalid obstacle placement.")

        # Add the obstacle
        self.board[y2][x2] = 3

        # Switch the current player
        self.current_player = 3 - self.current_player

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
        md5.update(board_str.encode("utf-8"))
        board_hash = md5.hexdigest()
        return board_hash

    def dump(self, add_param):
        file_name = os.path.join("data", self.hash_board())
        content = ""

        try:
            with open(file_name, "r") as f:
                content = f.read()
        except FileNotFoundError:
            pass

        if content:
            last_line = content.strip().split("\n")[-1]
            if last_line != add_param:
                print("Warning: The existing file has a different last line.")
            else:
                raise ValueError("Situation already exists")

        with open(file_name, "w") as f:
            content = self.getDataStr() + "\n" + add_param
            f.write(content)

        self.add(add_param)

        return True


def run_bots(bot1_path, bot2_path):
    chessboard = Chessboard()
    bots = [bot1_path, bot2_path]
    moves = ["-1 -1 -1 -1 -1 -1"]

    game_over = False
    turn = 0

    while not game_over:
        current_bot = subprocess.Popen(
            bots[turn % 2],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )

        if not current_bot.stdin or not current_bot.stdout:
            raise Exception("Current bot has no stdin or stdout")

        if turn % 2 == 0:
            history = "\n".join(moves) + "\n"
        else:
            history = "\n".join(moves[1:]) + "\n"

        input = f"{turn//2 + 1}\n{history}"

        os.system("cls")
        print(input)

        current_bot.stdin.write(input)
        current_bot.stdin.flush()

        move = current_bot.stdout.readline().strip()
        current_bot.kill()

        print(move)

        if move == "-1 -1 -1 -1 -1 -1":
            game_over = True
        else:
            moves.append(move)
            notExisted = chessboard.dump(move)
            # game_over = not notExisted


        turn += 1


if __name__ == "__main__":
    bot1_exe = "wuwuwu.exe"
    bot2_exe = "wuwuwu.exe"

    run_bots(bot1_exe, bot2_exe)