from glob import glob
import json
import os
import pickle
import numpy as np
import torch
from tqdm import tqdm
from model import NeuralNet
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
from pprint import pprint
import config


class MatchDataSet(Dataset):
    def __init__(self, data: dict):
        self.data = data
        self.keys = list(data.keys())

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        board = torch.tensor(self.data[self.keys[index]][0], dtype=torch.int8)
        answer = torch.tensor(self.data[self.keys[index]][1], dtype=torch.int8)
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
        input.append(
            NeuralNet.process_input(new_board, player, 2, answer[2], answer[3])
        )

        rand_val = torch.randn(3)

        p1, v1 = NeuralNet.process_output(
            rand_val[0], answer[0], answer[1], board, player, 0
        )

        p2, v2 = NeuralNet.process_output(
            rand_val[1], answer[2], answer[3], board, player, 1, answer[0], answer[1]
        )

        p3, v3 = NeuralNet.process_output(
            rand_val[2],
            answer[4],
            answer[5],
            new_board,
            player,
            2,
            answer[2],
            answer[3],
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
    return input[0:], (policies[0:], values[0:])


def load_dataset(path):
    input_paths = glob(path, recursive=False)

    data = {}
    for path in tqdm(input_paths, desc="Loading data", unit="file"):
        data = {**data, **pickle.load(open(path, "rb"))}

    return MatchDataSet(data)


if __name__ == "__main__":
    model_name = "model_09_40x128.pt"
    data_path = "data/augment/train/*.pickle"

    config.init()

    # load data
    train_dataset = load_dataset(data_path)
    train_loader = DataLoader(
        train_dataset, batch_size=config.batch_size, shuffle=True, collate_fn=collate_fn
    )

    # load model
    net = NeuralNet()
    if os.path.exists(model_name):
        net.load(path=model_name)

    # train
    optimizer = torch.optim.Adam(net.model.parameters(), lr=config.lr)
    loss_pi = nn.CrossEntropyLoss()
    loss_v = nn.MSELoss()
    net.model.train()

    for epoch in range(config.epochs):
        train_correct = 0
        train_actual_correct = 0
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

            corrects = out_pi.view(sample_size, -1).argmax(1) == target_pis.view(
                sample_size, -1
            ).argmax(1)
            # update correct
            train_correct += corrects.sum().item()

            # calculate actual correct: 3 consecutive corrects
            ac = 0
            for j in range(0, sample_size, 3):
                if corrects[j] and corrects[j + 1] and corrects[j + 2]:
                    ac += 1
            train_actual_correct += ac

            train_count += sample_size

            # backprop
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

            # update log
            if i % 10 == 0:
                config.run["policy loss"].append(train_ploss_sum / train_count)
                # config.run["value loss"].append(train_vloss_sum / train_count)
                config.run["accuracy"].append(train_correct / train_count)
                config.run["actual accuracy"].append(
                    train_actual_correct / (train_count / 3)
                )

                train_ploss_sum = 0
                train_vloss_sum = 0
                train_correct = 0
                train_actual_correct = 0
                train_count = 0

            if i % 50 == 0:
                net.save(model_name)
