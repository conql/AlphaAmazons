from glob import glob
import hashlib
import os
import pickle
import random
import numpy
import game
from tqdm import tqdm


def md5(data):
    return hashlib.md5(data).hexdigest()


def horizontal_flip(board: numpy.ndarray, answer):
    board2 = numpy.fliplr(board)
    answer2 = (
        7 - answer[0],
        answer[1],
        7 - answer[2],
        answer[3],
        7 - answer[4],
        answer[5],
    )

    return [(board, answer), (board2, answer2)]


def vertical_flip(board: numpy.ndarray, answer):
    board2 = numpy.flipud(board)
    answer2 = (
        answer[0],
        7 - answer[1],
        answer[2],
        7 - answer[3],
        answer[4],
        7 - answer[5],
    )

    return [(board, answer), (board2, answer2)]


def rotate(board: numpy.ndarray, answer):
    board2 = numpy.rot90(board, k=1)
    answer2 = (
        answer[1],
        7 - answer[0],
        answer[3],
        7 - answer[2],
        answer[5],
        7 - answer[4],
    )

    board3 = numpy.rot90(board, k=2)
    answer3 = (
        7 - answer[0],
        7 - answer[1],
        7 - answer[2],
        7 - answer[3],
        7 - answer[4],
        7 - answer[5],
    )

    board4 = numpy.rot90(board, k=3)
    answer4 = (
        7 - answer[1],
        answer[0],
        7 - answer[3],
        answer[2],
        7 - answer[5],
        answer[4],
    )

    return [(board, answer), (board2, answer2), (board3, answer3), (board4, answer4)]


def augment(func, data: list[tuple[numpy.ndarray, tuple]]):
    new_data = []
    for board, answer in data:
        new_data.extend(func(board, answer))

    return new_data


if __name__ == "__main__":
    input_path = "data/init/*.pickle"
    output_path = "data/augment3"
    multiplier = 0

    if not os.path.exists(output_path):
        os.makedirs(output_path)

    input_paths = glob(input_path, recursive=False)

    data = {}
    for path in input_paths:
        data = {**data, **pickle.load(open(path, "rb"))}

    new_data = {}

    for key, round in tqdm(data.items(), unit="round"):
        rounds = [round]
        rounds = augment(horizontal_flip, rounds)
        rounds = augment(vertical_flip, rounds)
        rounds = augment(rotate, rounds)

        # # validate all rounds
        # for board, ans in rounds:
        #     player = board[ans[1], ans[0]]
        #     assert player == 1 or player == 2

        #     if not game.is_valid_act(board, player, *ans):
        #         raise Exception(f"Invalid act: '{ans}'\n" + str(board))

        if multiplier != 0:
            add = random.sample(rounds, k=multiplier)
        else:
            add = rounds
        for board, ans in add:
            new_data[md5(board.tobytes())] = (board, ans)

    print(f"Original size: {len(data)}")
    print(f"Augmented size: {len(new_data)}")

    # split into several files
    for i in tqdm(range(0, len(new_data), 10000), unit="file", desc="Saving"):
        pickle.dump(
            dict(list(new_data.items())[i : i + 10000]),
            open(os.path.join(output_path, f"{i // 10000:03d}.pickle"), "wb"),
        )
