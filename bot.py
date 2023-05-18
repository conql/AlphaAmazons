from pprint import pprint
import torch
from torch import nn
import torch.nn.functional as F

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


class ResNet(nn.Module):
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


class Chessboard:
    def __init__(self):
        self.history = []
        self.board = torch.zeros((boardH, boardW), dtype=torch.int)
        self.init_board()
        self.current_player = 1

    def init_board(self):
        black_positions = [(0, 2), (2, 0), (5, 0), (7, 2)]
        white_positions = [(0, 5), (2, 7), (5, 7), (7, 5)]

        for position in black_positions:
            x, y = position
            self.board[y, x] = 1

        for position in white_positions:
            x, y = position
            self.board[y, x] = 2

    def is_valid_move(self, x0, y0, x1, y1):
        dx, dy = abs(x1 - x0), abs(y1 - y0)
        if self.board[y0, x0] != self.current_player or self.board[y1, x1] != 0:
            return False
        if dx == dy or dx == 0 or dy == 0:
            step_x = (x1 - x0) // max(1, abs(x1 - x0))
            step_y = (y1 - y0) // max(1, abs(y1 - y0))
            x, y = x0 + step_x, y0 + step_y
            while x != x1 or y != y1:
                if self.board[y, x] != 0:
                    return False
                x += step_x
                y += step_y
            return True
        return False

    def is_valid_obstacle(self, x1, y1, x2, y2):
        dx, dy = abs(x2 - x1), abs(y2 - y1)
        if self.board[y2, x2] != 0:
            return False
        if dx == dy or dx == 0 or dy == 0:
            step_x = (x2 - x1) // max(1, abs(x2 - x1))
            step_y = (y2 - y1) // max(1, abs(y2 - y1))
            x, y = x1 + step_x, y1 + step_y
            while x != x2 or y != y2:
                if self.board[y, x] != 0:
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
        self.board[y0, x0] = 0
        self.board[y1, x1] = self.current_player

        if not self.is_valid_obstacle(x1, y1, x2, y2):
            self.board[y0, x0] = self.current_player
            self.board[y1, x1] = 0

            # pprint(self.getData())
            raise ValueError("Invalid obstacle placement.")

        # Add the obstacle
        self.board[y2, x2] = 3

        # Add the move to history
        self.history.append(response)

        # Switch the current player
        self.current_player = 3 - self.current_player

    def get_valid_moves(self, player, stage, preActionX=None, preActionY=None):
        valid_moves = torch.zeros((8, 8))

        # Stage 0: Select chess
        if stage == 0:
            valid_moves[self.board == player] = 1
        # Stage 1: Move the selected chess
        elif stage == 1:
            for nx, ny in self.get_queen_moves(preActionX, preActionY):
                valid_moves[ny, nx] = 1
        # Stage 2: Place obstacle
        elif stage == 2:
            for nx, ny in self.get_queen_moves(preActionX, preActionY):
                valid_moves[ny, nx] = 1
        else:
            raise ValueError("Invalid stage")

        return valid_moves

    @staticmethod
    def is_on_board(x, y):
        return 0 <= x < 8 and 0 <= y < 8

    def get_queen_moves(self, x, y):
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
            while Chessboard.is_on_board(nx, ny) and self.board[ny][nx] == 0:
                moves.append((nx, ny))
                nx, ny = nx + dx, ny + dy
        return moves


# stage: 0 for piece selection
#        1 for piece movement
#        2 for obstacle placement
# player: 1 for player 1
#         2 for player 2
def process_input(board, player, stage, preActionX=None, preActionY=None):
    # data  [0, ] self chess
    #       [1, ] opponent chess
    #       [2, ] obstacles
    #       [3, ] selected piece
    #       [4, ] selected new move
    #       [5, ] stage 1 flag
    #       [6, ] stage 2 flag

    board = torch.tensor(board)

    data = torch.zeros((7, 8, 8))

    data[0, :, :] = (board == player).float()
    data[1, :, :] = (board == 3 - player).float()
    data[2, :, :] = (board == 3).float()

    if stage == 1:
        if board[preActionY, preActionX] != player:
            # pprint(board)
            raise ValueError("Moving non-player piece")
        data[3, preActionY, preActionX] = 1
        data[5, :, :] = 1
    elif stage == 2:
        if board[preActionY, preActionX] != player:
            # pprint(board)
            raise ValueError("Placing obstacle from non-player piece")
        data[4, preActionY, preActionX] = 1
        data[6, :, :] = 1

    # swap first two channels if playing as player 2
    if player != 1:
        data[[0, 1], ...] = data[[1, 0], ...]

    return data


def predict(model, chessboard: Chessboard, player, stage, preActionX=None, preActionY=None):
    board = chessboard.board
    input_data = process_input(board, player, stage, preActionX, preActionY)
    input_data = input_data.unsqueeze(0)

    with torch.no_grad():
        policy, value = model(input_data)

    policy = policy[0]
    policy_prob = F.softmax(policy.view(-1), dim=0).view(policy.shape)

    pprint(policy_prob.log10().round(decimals=2))

    # mask invalid moves
    valid_moves = chessboard.get_valid_moves(player, stage, preActionX, preActionY)
    policy_prob = policy_prob * valid_moves

    if policy_prob.sum() == 0:
        policy_prob = valid_moves * torch.rand_like(valid_moves).abs()

    # normalize
    policy_prob = policy_prob / policy_prob.sum()

    answer = policy_prob.view(-1).argmax().item()

    return (answer % 8, answer // 8)


def load_model(model_path):
    model = ResNet(40, 128)
    model.load_state_dict(torch.load(model_path, map_location=torch.device("cpu")))
    model.eval()
    return model


if __name__ == "__main__":
    model = load_model("data/model_07_40x128.pt")
    chessboard = Chessboard()

    round = int(input())
    player = 2
    for _ in range(1, round * 2):
        line = input().strip()
        if line == "-1 -1 -1 -1 -1 -1":
            player = 1
            continue
        chessboard.add(line)

    pieceX, pieceY = predict(model, chessboard, player, 0)
    moveX, moveY = predict(model, chessboard, player, 1, pieceX, pieceY)
    chessboard.board[pieceY, pieceX] = 0
    chessboard.board[moveY, moveX] = player
    obstacleX, obstacleY = predict(model, chessboard, player, 2, moveX, moveY)

    print(pieceX, pieceY, moveX, moveY, obstacleX, obstacleY)
