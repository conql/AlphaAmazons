from preprocessing import Chessboard
from pprint import pprint
chessboard = Chessboard()
with open("test.in", "r") as f:
    lines = f.readlines()[1:]
    for line in lines:
        chessboard.add(line)

pprint(chessboard.board)