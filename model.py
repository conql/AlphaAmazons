import os
from pprint import pprint
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
import config
import game

boardH = 8
boardW = 8
in_channel = 7


class CNNLayer(nn.Module):
    def __init__(self, in_channel, out_channel):
        super().__init__()
        self.conv = nn.Conv2d(
            in_channel,
            out_channel,
            3,
            stride=1,
            padding=1,
            dilation=1,
            groups=1,
            bias=False,
            padding_mode="zeros",
        )
        self.bn = nn.BatchNorm2d(out_channel)

    def forward(self, x):
        y = self.conv(x)
        y = self.bn(y)
        y = torch.relu(y)
        return y


class ResnetLayer(nn.Module):
    def __init__(self, inout_c, mid_c):
        super().__init__()
        self.conv_net = nn.Sequential(
            CNNLayer(inout_c, mid_c), CNNLayer(mid_c, inout_c)
        )

    def forward(self, x):
        x = self.conv_net(x) + x
        return x


class Outputhead(nn.Module):
    def __init__(self, out_c, head_mid_c):
        super().__init__()
        self.cnn = CNNLayer(out_c, head_mid_c)
        self.valueHeadLinear = nn.Linear(head_mid_c, 1)
        self.policyHeadLinear = nn.Conv2d(head_mid_c, 1, 1)

    def forward(self, h):
        x = self.cnn(h)

        # value head
        value = x.mean((2, 3))
        value = self.valueHeadLinear(value)

        # policy head
        policy = self.policyHeadLinear(x)
        policy = policy.squeeze(1)

        return policy, value


class Resnet(nn.Module):
    def __init__(self, b, f):
        super().__init__()
        self.model_name = "res"
        self.model_size = (b, f)

        self.inputhead = CNNLayer(in_channel, f)
        self.trunk = nn.ModuleList()
        for i in range(b):
            self.trunk.append(ResnetLayer(f, f))
        self.outputhead = Outputhead(f, f)

    def forward(self, x):
        h = self.inputhead(x)

        for block in self.trunk:
            h = block(h)

        return self.outputhead(h)


class NeuralNet:
    def __init__(self):
        self.model = Resnet(40, 128)

        if config.use_gpu:
            self.model.cuda()

    # stage: 0 for piece selection
    #        1 for piece movement
    #        2 for obstacle placement
    # player: 1 for player 1
    #         2 for player 2
    @staticmethod
    def process_input(board, player, stage, preActionX=None, preActionY=None):
        # data  [0, ] self chess
        #       [1, ] opponent chess
        #       [2, ] obstacles
        #       [3, ] selected piece
        #       [4, ] selected new move
        #       [5, ] stage 1 flag
        #       [6, ] stage 2 flag

        board = torch.tensor(board)
        if config.use_gpu:
            board = board.cuda()

        data = torch.zeros((7, 8, 8))

        assert player in [1, 2]

        data[0, :, :] = (board == player).float()
        data[1, :, :] = (board == 3 - player).float()
        data[2, :, :] = (board == 3).float()

        if stage == 1:
            if board[preActionY, preActionX] != player:
                pprint(board)
                raise ValueError("Moving non-player piece")
            data[3, preActionY, preActionX] = 1
            data[5, :, :] = 1
        elif stage == 2:
            if board[preActionY, preActionX] != player:
                pprint(board)
                raise ValueError("Placing obstacle from non-player piece")
            data[4, preActionY, preActionX] = 1
            data[6, :, :] = 1

        if config.use_gpu:
            data = data.cuda()

        return data

    @staticmethod
    def process_output(
        value, answerX, answerY, board, player, stage, preActionX=None, preActionY=None
    ):
        # valid_moves = game.get_valid_moves(board, player, stage, preActionX, preActionY)
        # valid_moves = torch.tensor(valid_moves)
        # if config.use_gpu:
        #     valid_moves = valid_moves.cuda()

        policy = torch.zeros((8, 8))
        # policy[valid_moves == 1] = 0.0001
        # policy[valid_moves == 0] = 0

        # if policy[answerY, answerX] == 0:
        #     pprint(board)
        #     raise ValueError("Invalid move")

        policy[answerY, answerX] = 1

        if config.use_gpu:
            policy = policy.cuda()

        return policy, value

    def predict(self, board, player, stage, preActionX=None, preActionY=None):
        input_data = self.process_input(board, player, stage, preActionX, preActionY)
        input_data = input_data.unsqueeze(0)

        self.model.eval()
        with torch.no_grad():
            policy, value = self.model(input_data)

        policy_prob = F.softmax(policy.view(policy.shape[0], -1), dim=1).view(
            policy.shape
        )
        value = torch.tanh(value)

        return policy_prob[0].cpu().numpy(), value

    def train(self, examples):
        optimizer = torch.optim.Adam(self.model.parameters(), lr=config.lr)

        for epoch in range(config.epochs):
            self.model.train()
            batch_count = int(len(examples) / config.batch_size)

            t = tqdm(range(batch_count), desc=f"Epoch {epoch + 1}/{config.epochs}")
            for _ in t:
                sample_ids = np.random.randint(len(examples), size=config.batch_size)
                (board, player, stage, preActionX, preActionY), pis, vs = list(
                    zip(*[examples[i] for i in sample_ids])
                )
                input_data = self.process_input(
                    board, player, stage, preActionX, preActionY
                )
                target_pis = torch.FloatTensor(np.array(pis))
                target_vs = torch.FloatTensor(np.array(vs).astype(np.float64))

                if config.use_gpu:
                    input_data, target_pis, target_vs = (
                        input_data.contiguous().cuda(),
                        target_pis.contiguous().cuda(),
                        target_vs.contiguous().cuda(),
                    )

                # predict
                out_pi, out_v = self.model(input_data)

                # calculate loss
                loss_pi = nn.CrossEntropyLoss(target_pis, out_pi)()
                loss_v = nn.MSELoss(target_vs, out_v)()
                total_loss = loss_pi + loss_v

                # update loss
                config.run["loss_pi"].append(loss_pi.item())
                config.run["loss_v"].append(loss_v.item())

                # backprop
                optimizer.zero_grad()
                total_loss.backward()
                optimizer.step()

    def save(self, path="model.pth"):
        folder = os.path.dirname(path)
        if not os.path.exists(folder) and folder != "":
            os.makedirs(folder)
        torch.save(self.model.state_dict(), path)

    def load(self, path="model.pth"):
        map_location = None if config.use_gpu else "cpu"
        self.model.load_state_dict(torch.load(path, map_location=map_location))
