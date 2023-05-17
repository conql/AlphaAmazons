import json
from pprint import pprint
import random
import lmdb
from preprocessing import Chessboard


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

generate_test_and_train()