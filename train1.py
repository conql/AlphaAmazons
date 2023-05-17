from pprint import pprint
import torch
from torch.utils.data import Dataset, DataLoader
import lmdb
import json


class MatchDataSet(Dataset):
    def __init__(self, db_path):
        self.db_path = db_path

        self.env = None
        self.txn = None

    def _init_db(self):
        self.env = lmdb.open(
            path=self.db_path,
            readonly=True,
            lock=False,
            readahead=False,
            meminit=False,
            map_size=2**30,
        )
        self.txn = self.env.begin()
        self.keys = [key for key, _ in self.txn.cursor()]

    def read_lmdb(self, key):
        lmdb_data = self.txn.get(key)  # type: ignore

        return lmdb_data.decode()

    def __len__(self):
        if self.env is None:
            self._init_db()

        return len(self.keys)

    def __getitem__(self, index):
        if self.env is None:
            self._init_db()

        lmdb_data = json.loads(self.read_lmdb(self.keys[index]))
        lmdb_data[1] = [int(x) for x in lmdb_data[1].split(" ")]
        match = torch.tensor(lmdb_data[0])
        answer = torch.tensor(lmdb_data[1])

        return match, answer


def is_on_board(x, y):
    return 0 <= x < 8 and 0 <= y < 8


def get_queen_moves(board, x, y):
    moves = []
    for dx, dy in [
        (-1, -1),
        (-1, 0),
        (-1, 1),
        (0, -1),
        (0, 1),
        (1, -1),
        (1, 0),
        (1, 1),
    ]:
        nx, ny = x + dx, y + dy
        while is_on_board(nx, ny) and board[nx, ny] == 0:
            moves.append((nx, ny))
            nx, ny = nx + dx, ny + dy
        if is_on_board(nx, ny) and board[nx, ny] != 0:
            break
    return moves


def get_valid_moves(board, answer, player):
    valid_moves = [torch.zeros((8, 8)) for _ in range(3)]

    # Stage 1: Select chess
    for x in range(8):
        for y in range(8):
            if board[x, y] == player:
                valid_moves[0][x, y] = 1

    # Stage 2: Move the selected chess
    x, y = answer[0].item(), answer[1].item()
    for nx, ny in get_queen_moves(board, x, y):
        valid_moves[1][nx, ny] = 1

    # Stage 3: Place obstacle
    x, y = answer[2].item(), answer[3].item()
    for nx, ny in get_queen_moves(board, x, y):
        valid_moves[2][nx, ny] = 1

    return valid_moves


def convert_to_train_data(match, answer):
    # match is a [8,8] tensor, in which 0 denotes empty, 1 denotes black, 2 denotes white, 3 denotes obstacles
    # count the number of black and white to determine the current player
    # answer is a [6] tensor, all of them are 0-7 integers
    # stage 0: select chess
    # stage 1: select new move
    # stage 2: select obstacle
    # answer[0] is the selected chess x
    # answer[1] is the selected chess y
    # answer[2] is the new move x
    # answer[3] is the new move y
    # answer[4] is obstacle x
    # answer[5] is obstacle y
    # output is a [8,8] tensor, representing the possibility of for each position in a stage
    # input is a [7, 8, 8] tensor
    
    # input[0] is my chesses
    # input[1] is opponent's chesses
    # input[2] is obstacles
    # input[3] selected chess
    # input[4] selected new move
    # input[5] if in stage 1, set to all 1 else 0
    # input[6] if in stage 2, set to all 1 else 0

    # return [
    #   (input, output) # for stage 0
    #   (input, output) # for stage 1
    #   (input, output) # for stage 2
    # ]


    # Get current player
    current_player = match[answer[1]][answer[0]]
    opponent_player = 3 - current_player

    if current_player != 1 and current_player != 2:
        raise ValueError("Invalid current player")

    # Initialize the input tensors for the three stages
    input_tensors = [torch.zeros((7, 8, 8)) for _ in range(3)]
    for i in range(3):
        input_tensors[i][0] = match == current_player
        input_tensors[i][1] = match == opponent_player
        input_tensors[i][2] = match == 3

    # Initialize the output tensors for the three stages
    output_tensors = [torch.zeros((8, 8)) for _ in range(3)]

    # Set the output tensors based on the answer
    for i in range(3):
        output_tensors[i][answer[2 * i], answer[2 * i + 1]] = 1

    # Get valid moves for each stage
    valid_moves = get_valid_moves(match, answer, current_player)

    # Set the invalid moves in output tensor to -1
    for i in range(3):
        invalid_mask = valid_moves[i] == 0
        output_tensors[i][invalid_mask] = -1

    # Set the selected chess in the input tensor for stages 1 and 2
    input_tensors[1][3] = output_tensors[0]

    # Set the selected new move in the input tensor for stage 2
    input_tensors[2][4] = output_tensors[1]

    # Set the stage in the input tensor for stages 1 and 2
    input_tensors[1][5] = torch.ones((8, 8))
    input_tensors[2][6] = torch.ones((8, 8))

    return [
        (input_tensors[0], output_tensors[0]),  # for stage 0
        (input_tensors[1], output_tensors[1]),  # for stage 1
        (input_tensors[2], output_tensors[2]),  # for stage 2
    ]


if __name__ == "__main__":
    train_dataset = MatchDataSet("data/lmdb")
    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=False)

    for i, (X, Y) in enumerate(train_loader):
        input = convert_to_train_data(X[0], Y[0])
        pprint(X[0])
        pprint(input[0])
        break
