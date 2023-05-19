from glob import glob
import hashlib
import json
import os
from pprint import pprint
import random
import lmdb
from tqdm import tqdm
from preprocessing import Chessboard
import pickle
import numpy as np
import game


def validate_data_item(txn, key):
    lmdb_data = txn.get(key)  # type: ignore
    lmdb_data = json.loads(lmdb_data.decode())
    answer = [int(x) for x in lmdb_data[1].split(" ")]

    chessboard = Chessboard()
    chessboard.board = lmdb_data[0]
    chessboard.current_player = lmdb_data[0][answer[1]][answer[0]]
    try:
        chessboard.add(lmdb_data[1])
    except Exception as e:
        # pprint(lmdb_data)
        raise Exception(f"Invalid data at key {key}: {e}")


def validate_data(data_path="data/lmdb"):
    env = lmdb.open(data_path, readonly=True)
    invalid = 0
    total = 0
    with env.begin() as txn:
        for key, _ in txn.cursor():
            try:
                total += 1
                validate_data_item(txn, key)
            except Exception as e:
                invalid += 1
                # print(e)

    print(f"Total: {total}, invalid: {invalid}")


def display_random_items(db_path="data/lmdb", num_items=100):
    # Open the LMDB database
    env = lmdb.open(db_path, readonly=True)
    total_entries = env.stat()["entries"]

    if total_entries < num_items:
        print(
            f"Warning: The database has only {total_entries} entries. Displaying all."
        )
        num_items = total_entries
    else:
        print(f"Displaying first {num_items} entries. Total entries: {total_entries}")

    with env.begin() as txn:
        cursor = txn.cursor()
        count = 0

        for key, value in cursor:
            # Decode the key and value (assuming UTF-8 encoding)
            decoded_key = key.decode("utf-8")
            decoded_value = json.loads(value.decode("utf-8"))
            pprint(decoded_key)
            pprint(decoded_value)

            count += 1

            if count == num_items:
                break

    env.close()


def generate_test_and_train(db_path="data/lmdb", train_percent=0.9):
    env = lmdb.open(
        path=db_path,
        readonly=True,
        lock=False,
        readahead=False,
        meminit=False,
        map_size=2**29,
    )
    txn = env.begin()
    keys = [key.decode() for key, _ in txn.cursor()]
    random.shuffle(keys)
    train_idx = int(len(keys) * train_percent)
    train_keys = keys[:train_idx]
    test_keys = keys[train_idx:]

    with open("data/train_keys.txt", "w") as f:
        f.write("\n".join(train_keys))
        f.flush()
    with open("data/test_keys.txt", "w") as f:
        f.write("\n".join(test_keys))
        f.flush()

    env.close()


def data_augmentation(db_path="data/lmdb"):
    pass


def md5(data):
    return hashlib.md5(data).hexdigest()


def convert_lmdb_pickle(db_path="data/lmdb", output_dir="data/init"):
    env = lmdb.open(
        path=db_path,
        readonly=True,
        lock=False,
        readahead=False,
        meminit=False,
        map_size=2**29,
    )
    total_entries = env.stat()["entries"]

    all_datas = []
    current_block = {}

    with env.begin() as txn:
        cursor = txn.cursor()
        for key, value in tqdm(cursor, total=total_entries):
            key = key.decode()
            value = json.loads(value.decode())

            board = np.array(value[0], dtype=np.int8)
            move = np.array([int(x) for x in value[1].split(" ")], dtype=np.int8)

            current_player = board[move[1], move[0]]
            if current_player == 1:
                pass
            elif current_player == 2:
                # replace 1 with 2 and 2 with 1
                board[board == 1] = 6
                board[board == 2] = 1
                board[board == 6] = 2
            else:
                raise Exception(f"Invalid current player: {current_player}")

            new_key = md5(board.tobytes())

            current_block[new_key] = (board, move)

            if len(current_block) == 100000:
                all_datas.append(current_block)
                current_block = {}

    all_datas.append(current_block)

    # print data size
    print(f"Total data: {total_entries}, total blocks: {len(all_datas)}")
    print("Writing data to pickle")
    # with open("data/data.pickle", "wb") as f:
    #     pickle.dump(data, f)

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for i, data in enumerate(all_datas):
        with open(os.path.join(output_dir, f"block_{i:02d}.pickle"), "wb") as f:
            pickle.dump(data, f)
    print("Done")


def validate_data_2(input_path="data/init/*.pickle"):
    from preprocessing2 import get_answer

    input_paths = glob(input_path, recursive=False)

    data = {}
    for path in tqdm(input_paths, desc="Loading data", unit="file"):
        data = {**data, **pickle.load(open(path, "rb"))}

    keys = list(data.keys())
    random.shuffle(keys)

    wrong_count = 0
    for i, key in tqdm(enumerate(keys), desc="Verifying data"):
        (board, answer) = data[key]
        ac = get_answer(board, board[answer[1], answer[0]])

        if ac != list(answer):
            wrong_count += 1
            tqdm.write(f"{wrong_count}/{i} ({wrong_count/i*100:.02f})")


def validate_data_3(input_path="data/augment2/*.pickle"):
    from preprocessing2 import get_answer

    input_paths = glob(input_path, recursive=False)

    data = {}
    for path in tqdm(input_paths, desc="Loading data", unit="file"):
        data = {**data, **pickle.load(open(path, "rb"))}

    keys = list(data.keys())
    print(f"Total length: {len(keys)}")
    random.shuffle(keys)

    wrong_count = 0
    for i, key in tqdm(enumerate(keys), desc="Verifying data", total=len(keys)):
        (board, answer) = data[key]
        if not game.is_valid_act(board, board[answer[1], answer[0]], *answer):
            wrong_count += 1
            tqdm.write(f"{wrong_count}/{i} ({wrong_count/i*100:.02f})")

    tqdm.write(f"{wrong_count}/{len(keys)}")


if __name__ == "__main__":
    # validate_data()
    # display_random_items()
    # generate_test_and_train()
    # data_augmentation()
    # convert_lmdb_pickle()

    validate_data_3()
