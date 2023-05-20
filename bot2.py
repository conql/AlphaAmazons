# from pprint import pprint
import random
import time
import numpy
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


class game:
    @staticmethod
    def get_valid_moves(
        board: numpy.ndarray, player, stage, preActionX=None, preActionY=None
    ):
        valid_moves = numpy.zeros((8, 8), dtype=numpy.int8)
        # Stage 0: Select chess
        if stage == 0:
            valid_moves[board == player] = 1
        # Stage 1: Move the selected chess
        elif stage == 1:
            for nx, ny in game.get_queen_moves(board, preActionX, preActionY):
                valid_moves[ny, nx] = 1
        # Stage 2: Place obstacle
        elif stage == 2:
            for nx, ny in game.get_queen_moves(board, preActionX, preActionY):
                valid_moves[ny, nx] = 1
        else:
            raise ValueError("Invalid stage")

        return valid_moves

    @staticmethod
    def get_next_state(
        board: numpy.ndarray, player, stage, ax, ay, preActionX=None, preActionY=None
    ):
        if stage == 0:
            assert board[ay, ax] == player
            return (board, player, 1, ax, ay)
        else:
            new_board = board.copy()

            if stage == 1:
                assert board[preActionY, preActionX] == player
                new_board[preActionY, preActionX] = 0
                new_board[ay, ax] = player
                return (new_board, player, 2, ax, ay)
            elif stage == 2:
                assert board[preActionY, preActionX] == player
                new_board[ay, ax] = 3
                return (new_board, 3 - player, 0, None, None)
            else:
                raise ValueError("Invalid stage")

    @staticmethod
    def get_game_ended(
        board: numpy.ndarray, player, stage, preActionX=None, preActionY=None
    ):
        valid = game.get_valid_moves(board, player, stage, preActionX, preActionY)
        if valid.sum() == 0:
            return -1  # player loses
        else:
            return 0  # game continues

    @staticmethod
    def is_on_board(x, y):
        return 0 <= x < 8 and 0 <= y < 8

    @staticmethod
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
            while game.is_on_board(nx, ny) and board[ny, nx] == 0:
                moves.append((nx, ny))
                nx, ny = nx + dx, ny + dy
        return moves

    @staticmethod
    def get_state_representation(
        board: numpy.ndarray, player, stage, preActionX=None, preActionY=None
    ):
        board_hash = hash(tuple(board.flatten()))
        return (board_hash, player, stage, preActionX, preActionY)

    @staticmethod
    def is_valid_move(board, player, x0, y0, x1, y1):
        dx, dy = abs(x1 - x0), abs(y1 - y0)
        if board[y0, x0] != player or board[y1, x1] != 0:
            return False
        if dx == dy or dx == 0 or dy == 0:
            step_x = (x1 - x0) // max(1, abs(x1 - x0))
            step_y = (y1 - y0) // max(1, abs(y1 - y0))
            x, y = x0 + step_x, y0 + step_y
            while x != x1 or y != y1:
                if board[y, x] != 0:
                    return False
                x += step_x
                y += step_y
            return True
        return False

    @staticmethod
    def is_valid_obstacle(board, x1, y1, x2, y2):
        dx, dy = abs(x2 - x1), abs(y2 - y1)
        if board[y2, x2] != 0:
            return False
        if dx == dy or dx == 0 or dy == 0:
            step_x = (x2 - x1) // max(1, abs(x2 - x1))
            step_y = (y2 - y1) // max(1, abs(y2 - y1))
            x, y = x1 + step_x, y1 + step_y
            while x != x2 or y != y2:
                if board[y, x] != 0:
                    return False
                x += step_x
                y += step_y
            return True
        return False

    @staticmethod
    def is_valid_act(board, player, x0, y0, x1, y1, x2, y2):
        if player != 1 and player != 2:
            return False

        board = board.copy()
        if game.is_valid_move(board, player, x0, y0, x1, y1):
            board[y0, x0] = 0
            board[y1, x1] = player

            if game.is_valid_obstacle(board, x1, y1, x2, y2):
                return True
        else:
            return False

    @staticmethod
    def take_act(board, player, response):
        x0, y0, x1, y1, x2, y2 = map(int, response.split())

        if not game.is_valid_move(board, player, x0, y0, x1, y1):
            # pprint(self.getData())
            raise ValueError("Invalid move.")

        # Move the piece
        board[y0, x0] = 0
        board[y1, x1] = player

        if not game.is_valid_obstacle(board, x1, y1, x2, y2):
            board[y0, x0] = player
            board[y1, x1] = 0

            # pprint(self.getData())
            raise ValueError("Invalid obstacle placement.")

        # Add the obstacle
        board[y2, x2] = 3

    @staticmethod
    def init_board():
        board = numpy.zeros((8, 8), dtype=numpy.int8)
        black_positions = [(0, 2), (2, 0), (5, 0), (7, 2)]
        white_positions = [(0, 5), (2, 7), (5, 7), (7, 5)]

        for position in black_positions:
            x, y = position
            board[y, x] = 1

        for position in white_positions:
            x, y = position
            board[y, x] = 2

        return board


class Chessboard:
    def __init__(self):
        self.history = []
        self.board = game.init_board()
        self.current_player = 1

    def add(self, response):
        game.take_act(self.board, self.current_player, response)
        # Add the move to history
        self.history.append(response)
        # Switch the current player
        self.current_player = 3 - self.current_player


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

    return data


def predict(model, board, player, stage, preActionX=None, preActionY=None):
    input_data = process_input(board, player, stage, preActionX, preActionY)
    input_data = input_data.unsqueeze(0)

    with torch.no_grad():
        policy, value = model(input_data)

    policy = policy[0]
    policy_prob = F.softmax(policy.view(-1), dim=0).view(policy.shape)

    # pprint(policy_prob.log10().round(decimals=2))

    # mask invalid moves
    valid_moves = torch.tensor(
        game.get_valid_moves(chessboard.board, player, stage, preActionX, preActionY)
    )
    policy_prob = policy_prob * valid_moves

    if policy_prob.sum() == 0:
        policy_prob = valid_moves * torch.rand_like(valid_moves).abs()

    # normalize
    policy_prob = policy_prob / policy_prob.sum()

    return (policy_prob.numpy(), value.item())


def load_model(model_path):
    model = ResNet(40, 128)
    model.load_state_dict(torch.load(model_path, map_location=torch.device("cpu")))
    model.eval()
    return model


class MCTS:
    def __init__(self, net: ResNet, c_puct=5, simulate_times=100):
        self.net = net
        self.c_puct = c_puct
        self.simulate_times = simulate_times

        self.Qsa = {}  # stores Q values for s,a (as defined in the paper)
        self.Nsa = {}  # stores #times edge s,a was visited
        self.Ns = {}  # stores #times board s was visited
        self.Ps = {}  # stores initial policy (returned by neural net)

        self.Es = {}  # stores game.getGameEnded ended for board s
        self.Vs = {}  # stores game.getValidMoves for board s

    def search(self, board, player, stage, preActionX=None, preActionY=None):
        s = game.get_state_representation(board, player, stage, preActionX, preActionY)

        if s not in self.Es:
            self.Es[s] = game.get_game_ended(
                board, player, stage, preActionX, preActionY
            )

        if self.Es[s] != 0:
            # terminal node
            return stage == 0 if -self.Es[s] else self.Es[s]

        if s not in self.Ps:
            # leaf node
            self.Ps[s], v = predict(
                self.net, board, player, stage, preActionX, preActionY
            )
            valid_moves = game.get_valid_moves(
                board, player, stage, preActionX, preActionY
            )
            self.Ps[s] = self.Ps[s] * valid_moves  # masking invalid moves
            sum_Ps_s = numpy.sum(self.Ps[s])

            if sum_Ps_s > 0:
                # renormalize
                self.Ps[s] /= sum_Ps_s
            else:
                # all valid moves were masked
                # print("All valid moves were masked, doing a workaround.")
                self.Ps[s] = self.Ps[s] + valid_moves
                self.Ps[s] /= numpy.sum(self.Ps[s])

            self.Vs[s] = valid_moves
            self.Ns[s] = 0

            return stage == 0 if -self.Es[s] else self.Es[s]

        valid_moves = self.Vs[s]
        max_u = -float("inf")
        best_a = None

        # pick the action with the highest upper confidence bound
        for ay in range(8):
            for ax in range(8):
                if valid_moves[ay, ax]:
                    if (s, (ax, ay)) in self.Qsa:  # Qsa computed?
                        u = self.Qsa[(s, (ax, ay))] + self.c_puct * self.Ps[s][
                            ay, ax
                        ] * numpy.sqrt(self.Ns[s]) / (1 + self.Nsa[(s, (ax, ay))])
                    else:
                        u = (
                            self.c_puct
                            * self.Ps[s][ay, ax]
                            * numpy.sqrt(self.Ns[s] + 1e-8)
                        )

                    if u > max_u:
                        max_u = u
                        best_a = (ax, ay)

        assert best_a is not None
        a = best_a

        v = self.search(
            *game.get_next_state(
                board, player, stage, a[0], a[1], preActionX, preActionY
            )
        )

        if (s, a) in self.Qsa:
            self.Qsa[(s, a)] = (self.Nsa[(s, a)] * self.Qsa[(s, a)] + v) / (
                self.Nsa[(s, a)] + 1
            )  # update Qsa
            self.Nsa[(s, a)] += 1  # update Nsa

        else:
            # initialize Qsa and Nsa
            self.Qsa[(s, a)] = v
            self.Nsa[(s, a)] = 1

        self.Ns[s] += 1
        return stage == 0 if -v else v

    def compute_policy(
        self, temperature, board, player, stage, preActionX=None, preActionY=None
    ):
        for _ in range(self.simulate_times):
            self.search(board, player, stage, preActionX, preActionY)

        s = game.get_state_representation(board, player, stage, preActionX, preActionY)

        na = numpy.zeros((8, 8))
        for ay in range(8):
            for ax in range(8):
                na[ay, ax] = self.Nsa[(s, (ax, ay))] if (s, (ax, ay)) in self.Nsa else 0

        if temperature == 0:
            max_n = numpy.max(na)
            best_as = numpy.nonzero(na == max_n)
            a = random.choice(best_as)
            policy = numpy.zeros((8, 8))
            policy[a[0], a[1]] = 1
        else:
            na = numpy.power(na, 1 / temperature)
            policy = na / numpy.sum(na)
        return policy

    def predict(self, board, player, stage, preActionX=None, preActionY=None):
        policy = self.compute_policy(1, board, player, stage, preActionX, preActionY)
        answer = policy.flatten().argmax().item()
        return (answer % 8, answer // 8)


if __name__ == "__main__":
    # use time
    start = time.time()

    model = load_model("data/resnet_01_40x128.pt")
    chessboard = Chessboard()

    round = int(input())
    player = 2
    for _ in range(1, round * 2):
        line = input().strip()
        if line == "-1 -1 -1 -1 -1 -1":
            player = 1
            continue
        chessboard.add(line)

    mcts = MCTS(model, 5, 25)

    pieceX, pieceY = mcts.predict(chessboard.board, player, 0)
    moveX, moveY = mcts.predict(chessboard.board, player, 1, pieceX, pieceY)
    chessboard.board[pieceY, pieceX] = 0
    chessboard.board[moveY, moveX] = player
    obstacleX, obstacleY = mcts.predict(chessboard.board, player, 2, moveX, moveY)

    print(pieceX, pieceY, moveX, moveY, obstacleX, obstacleY)
    # print(value1.item(), value2.item(), value3.item())

    print("time:", time.time() - start)
