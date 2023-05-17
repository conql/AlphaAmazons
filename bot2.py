import torch
import game
from game import Chessboard
from MCTS import MCTS
from model import NeuralNet
import sys

if __name__ == "__main__":
    net = NeuralNet()
    net.load(path="model_05_40x128.pt")
    mcts = MCTS(net)
    chessboard = Chessboard()

    round = int(input())
    for _ in range(1, round * 2):
        request = input().strip()
        if request != "-1 -1 -1 -1 -1 -1":
            chessboard.act(request)

    pieceX, pieceY = mcts.predict(chessboard.board, chessboard.player, 0)
    moveX, moveY = mcts.predict(chessboard.board, chessboard.player, 1, pieceX, pieceY)
    chessboard.board[pieceY, pieceX] = 0
    chessboard.board[moveY, moveX] = chessboard.player
    obstacleX, obstacleY = mcts.predict(chessboard.board, chessboard.player, 2, moveX, moveY)

    print(pieceX, pieceY, moveX, moveY, obstacleX, obstacleY)

