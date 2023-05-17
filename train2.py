import json
import os
import lmdb
import numpy as np
import torch
from tqdm import tqdm
from model import NeuralNet
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
from pprint import pprint
import config


class MatchDataSet(Dataset):
    def __init__(self, db_path, keys):
        self.db_path = db_path
        self.env = None
        self.txn = None
        self.keys = keys

    def _init_db(self):
        self.env = lmdb.open(
            path=self.db_path,
            readonly=True,
            lock=False,
            readahead=False,
            meminit=False,
            map_size=2**29,
        )
        self.txn = self.env.begin()

    def read_lmdb(self, key):
        lmdb_data = self.txn.get(key.encode())  # type: ignore

        return lmdb_data.decode()

    def __len__(self):
        return len(self.keys)

    def __getitem__(self, index):
        if self.env is None:
            self._init_db()

        lmdb_data = json.loads(self.read_lmdb(self.keys[index]))
        lmdb_data[1] = [int(x) for x in lmdb_data[1].split(" ")]
        board = torch.LongTensor(lmdb_data[0])
        answer = torch.LongTensor(lmdb_data[1])

        return (board, answer)


def collate_fn(batch):
    input = []
    policies = []
    values = []
    for i, (board, answer) in enumerate(batch):
        player = board[answer[1]][answer[0]]
        if player != 1 and player != 2:
            raise Exception("Invalid player: {}".format(player))

        # pprint(board)
        # pprint(answer)

        new_board = board.clone()
        new_board[answer[1]][answer[0]] = 0
        new_board[answer[3]][answer[2]] = player

        input.append(NeuralNet.process_input(board, player, 0))
        input.append(NeuralNet.process_input(board, player, 1, answer[0], answer[1]))
        input.append(NeuralNet.process_input(new_board, player, 2, answer[2], answer[3]))

        rand_val = torch.randn(3)

        p1, v1 = NeuralNet.process_output(
            rand_val[0], answer[0], answer[1], board, player, 0
        )

        p2, v2 = NeuralNet.process_output(
            rand_val[1], answer[2], answer[3], board, player, 1, answer[0], answer[1]
        )

        p3, v3 = NeuralNet.process_output(
            rand_val[2], answer[4], answer[5], new_board, player, 2, answer[2], answer[3]
        )

        policies.append(p1)
        policies.append(p2)
        policies.append(p3)

        values.append(v1)
        values.append(v2)
        values.append(v3)

    # convert to torch tensor
    input = torch.stack(input)
    policies = torch.stack(policies)
    values = torch.stack(values)
    return input, (policies, values)


if __name__ == "__main__":
    config.init()

    with open("data/train_keys.txt", "r") as f:
        train_keys = f.read().splitlines()

    train_dataset = MatchDataSet("data/lmdb", train_keys)
    train_loader = DataLoader(
        train_dataset, batch_size=config.batch_size, shuffle=True, collate_fn=collate_fn
    )

    net = NeuralNet()
    model_name = "model_05_40x128.pt"
    if os.path.exists(model_name):
        net.load(path=model_name)

    optimizer = torch.optim.Adam(net.model.parameters(), lr=config.lr)
    # calculate loss
    loss_pi = nn.CrossEntropyLoss()
    loss_v = nn.MSELoss()
    net.model.train()

    for epoch in range(config.epochs):
        train_correct = 0
        train_ploss_sum = 0
        train_vloss_sum = 0
        train_count = 0
        for i, (X, Y) in tqdm(
            enumerate(train_loader), total=len(train_loader), desc="Training model"
        ):
            input_data = X
            target_pis, target_vs = Y
            sample_size = input_data.shape[0]

            if config.use_gpu:
                input_data, target_pis, target_vs = (
                    input_data.contiguous().cuda(),
                    target_pis.contiguous().cuda(),
                    target_vs.contiguous().cuda(),
                )

            # predict
            out_pi, out_v = net.model(input_data)

            p_loss = loss_pi(out_pi, target_pis)
            # v_loss = loss_v(out_v, target_vs)

            # pprint(out_pi)
            # pprint(target_pis)

            total_loss = p_loss

            # update loss
            train_ploss_sum += p_loss.item() * sample_size
            # train_vloss_sum += v_loss.item() * sample_size

            # update correct
            train_correct += (
                (
                    out_pi.view(sample_size, -1).argmax(1)
                    == target_pis.view(sample_size, -1).argmax(1)
                )
                .sum()
                .item()
            )

            train_count += sample_size

            # backprop
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

            if i % 10 == 0:
                config.run["policy loss"].append(train_ploss_sum / train_count)
                # config.run["value loss"].append(train_vloss_sum / train_count)
                config.run["accuracy"].append(train_correct / train_count)

                train_ploss_sum = 0
                train_vloss_sum = 0
                train_correct = 0
                train_count = 0

            if i % 50 == 0:
                net.save(model_name)
