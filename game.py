from pprint import pprint
import numpy
import config


def get_valid_moves(board:numpy.ndarray, player, stage, preActionX=None, preActionY=None):
    valid_moves = numpy.zeros((8, 8), dtype=numpy.int8)
    # Stage 0: Select chess
    if stage == 0:
        valid_moves[board == player] = 1
    # Stage 1: Move the selected chess
    elif stage == 1:
        for nx, ny in get_queen_moves(board, preActionX, preActionY):
            valid_moves[ny, nx] = 1
    # Stage 2: Place obstacle
    elif stage == 2:
        for nx, ny in get_queen_moves(board, preActionX, preActionY):
            valid_moves[ny, nx] = 1
    else:
        raise ValueError("Invalid stage")

    return valid_moves


def get_next_state(board:numpy.ndarray, player, stage, ax, ay, preActionX=None, preActionY=None):
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


def get_game_ended(board:numpy.ndarray, player, stage, preActionX=None, preActionY=None):
    valid = get_valid_moves(board, player, stage, preActionX, preActionY)
    if valid.sum() == 0:
        return -1  # player loses
    else:
        return 0  # game continues


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
        while is_on_board(nx, ny) and board[ny, nx] == 0:
            moves.append((nx, ny))
            nx, ny = nx + dx, ny + dy
    return moves


def get_state_representation(
    board: numpy.ndarray, player, stage, preActionX=None, preActionY=None
):
    board_hash = hash(tuple(board.flatten()))
    return (board_hash, player, stage, preActionX, preActionY)


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


def take_act(board, player, response):
    x0, y0, x1, y1, x2, y2 = map(int, response.split())

    if not is_valid_move(board, player, x0, y0, x1, y1):
        # pprint(self.getData())
        raise ValueError("Invalid move.")

    # Move the piece
    board[y0, x0] = 0
    board[y1, x1] = player

    if not is_valid_obstacle(board, x1, y1, x2, y2):
        board[y0, x0] = player
        board[y1, x1] = 0

        # pprint(self.getData())
        raise ValueError("Invalid obstacle placement.")

    # Add the obstacle
    board[y2, x2] = 3


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
    def __init__(self) -> None:
        self.board = init_board()
        self.player = 1
        self.history = []

    def act(self, response):
        take_act(self.board, self.player, response)
        self.player = 3 - self.player
        self.history.append(response)