from math import sqrt
import torch
import torch.nn as nn
import numpy as np
import json


boardH = 8
boardW = 8
input_c = 7


class CNNLayer(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()
        self.conv = nn.Conv2d(
            in_c,
            out_c,
            3,
            stride=1,
            padding=1,
            dilation=1,
            groups=1,
            bias=False,
            padding_mode="zeros",
        )
        self.bn = nn.BatchNorm2d(out_c)

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


class Outputhead_v1(nn.Module):
    def __init__(self, out_c, head_mid_c):
        super().__init__()
        self.cnn = CNNLayer(out_c, head_mid_c)
        self.valueHeadLinear = nn.Linear(head_mid_c, 3)
        self.policyHeadLinear = nn.Conv2d(head_mid_c, 1, 1)

    def forward(self, h):
        x = self.cnn(h)

        # value head
        value = x.mean((2, 3))
        value = self.valueHeadLinear(value)

        # policy head
        policy = self.policyHeadLinear(x)
        policy = policy.squeeze(1)

        return value, policy


class Model_ResNet(nn.Module):
    def __init__(self, b, f):
        super().__init__()
        self.model_name = "res"
        self.model_size = (b, f)

        self.inputhead = CNNLayer(input_c, f)
        self.trunk = nn.ModuleList()
        for i in range(b):
            self.trunk.append(ResnetLayer(f, f))
        self.outputhead = Outputhead_v1(f, f)

    def forward(self, x):
        if x.shape[1] == 2:  # global feature is none
            x = torch.cat((x, torch.zeros((1, input_c - 2, boardH, boardW))), dim=1)
        h = self.inputhead(x)

        for block in self.trunk:
            h = block(h)

        return self.outputhead(h)


ModelDic = {"res": Model_ResNet}


class Board:
    def __init__(self, color):
        # 0:Me   1:Opp   2:banLoc
        self.board = np.zeros(shape=(3, boardH, boardW))
        self.stage = 0
        self.chosenX = -1
        self.chosenY = -1
        self.board[0, 0, 2] = 1
        self.board[0, 0, 5] = 1
        self.board[0, 2, 0] = 1
        self.board[0, 2, 7] = 1
        self.board[1, 5, 0] = 1
        self.board[1, 5, 7] = 1
        self.board[1, 7, 2] = 1
        self.board[1, 7, 5] = 1

        if color == 1:  # white, inverse color
            self.board[0, :, :] = self.board[0, :, :] + self.board[1, :, :]
            self.board[1, :, :] = self.board[0, :, :] - self.board[1, :, :]
            self.board[0, :, :] = self.board[0, :, :] - self.board[1, :, :]

    def playInit(self, x0, y0, x1, y1, x2, y2):
        self.board[:, y1, x1] = self.board[:, y0, x0]
        self.board[:, y0, x0] = 0
        self.board[2, y2, x2] = 1

    def playMe(self, x, y):
        if self.stage == 0:
            self.stage = 1
            self.chosenX = x
            self.chosenY = y
        elif self.stage == 1:
            self.stage = 2
            self.board[:, y, x] = self.board[:, self.chosenY, self.chosenX]
            self.board[:, self.chosenY, self.chosenX] = 0
            self.chosenX = x
            self.chosenY = y
        else:
            assert (False, "This is not necessary because there is no search")

    def getNNinput(self):
        nninput = np.zeros(shape=(7, boardH, boardW))
        nninput[0:3, :, :] = self.board
        if self.stage == 0:
            pass
        elif self.stage == 1:
            nninput[5, :, :] = 1
            nninput[3, self.chosenY, self.chosenX] = 1
        else:
            nninput[6, :, :] = 1
            nninput[4, self.chosenY, self.chosenX] = 1
        return nninput

    def hasStone(self, x, y):
        return (
            self.board[0, y, x] != 0
            or self.board[1, y, x] != 0
            or self.board[2, y, x] != 0
        )

    def isQueenMove(self, x0, y0, x1, y1):
        if not self.hasStone(x0, y0):
            return False
        if self.hasStone(x1, y1):
            return False
        if x0 == x1 and y0 == y1:
            return False
        dx = x1 - x0
        dy = y1 - y0
        if dx != 0 and dy != 0 and dx != dy and dx != -dy:
            return False

        d = max(abs(dx), abs(dy))
        if dx > 0:
            dx = 1
        elif dx < 0:
            dx = -1
        if dy > 0:
            dy = 1
        elif dy < 0:
            dy = -1
        for i in range(1, d):
            x = x0 + i * dx
            y = y0 + i * dy
            if self.hasStone(x, y):
                return False
        return True

    def isLegal(self, x, y):
        if self.stage == 0:
            return self.board[0, y, x] == 1
        elif self.stage == 1:
            return self.isQueenMove(self.chosenX, self.chosenY, x, y)
        else:
            return self.isQueenMove(self.chosenX, self.chosenY, x, y)


if __name__ == "__main__":
    file_path = "data/amazons_40x128.pth"
    # 这份代码并没有什么特别之处，就是调用了一个非常基础的resnet
    # 猜猜我的model怎么来的
    modeldata = torch.load(file_path, map_location=torch.device("cpu"))

    model_type = modeldata["model_name"]
    model_param = modeldata["model_size"]
    model = ModelDic[model_type](*model_param)
    model.load_state_dict(modeldata["state_dict"])
    model.eval()

    # 解析读入的JSON
    full_input = json.loads(input())
    # 分析自己收到的输入和自己过往的输出，并恢复状态
    all_requests = full_input["requests"]
    all_responses = full_input["responses"]

    color = 1  # default white
    last_request = all_requests[-1]
    if int(all_requests[0]["x0"]) == -1:
        # all_requests=all_requests[1:]
        color = 0  # black

    board = Board(color)
    for i in range(len(all_requests)):
        r = all_requests[i]
        if r["x0"] != -1:
            board.playInit(r["x0"], r["y0"], r["x1"], r["y1"], r["x2"], r["y2"])
        if i < len(all_responses):
            r = all_responses[i]
            board.playInit(r["x0"], r["y0"], r["x1"], r["y1"], r["x2"], r["y2"])

    # 神经网络计算 --------------------------------------------------------------------------------------------------------

    my_action = {}
    # stage0:选子   stage1:落子   stage2:放障碍
    for stage in range(3):
        nninput = torch.FloatTensor(board.getNNinput())
        nninput.unsqueeze_(0)
        v, p = model(nninput)

        movenum = len(all_requests) * 2
        policytemp = 0.5 * (0.5 ** (movenum / 10)) + 0.01

        policy = p.detach().numpy().reshape((-1))
        policy = policy - np.max(policy)
        for i in range(boardW * boardH):
            if not board.isLegal(i % boardW, i // boardW):
                policy[i] = -10000
        policy = policy - np.max(policy)
        for i in range(boardW * boardH):
            if policy[i] < -1:
                policy[i] = -10000
        policy = policy / policytemp
        probs = np.exp(policy)
        probs = probs / sum(probs)

        action = int(np.random.choice([i for i in range(boardW * boardW)], p=probs))
        preActionX = action % boardW
        preActionY = action // boardW
        board.playMe(preActionX, preActionY)
        my_action["x" + str(stage)] = preActionX
        my_action["y" + str(stage)] = preActionY


print(json.dumps({"response": my_action}))
