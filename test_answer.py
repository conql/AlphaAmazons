import os

from game import Chessboard

if __name__ == "__main__":
    
    chessboard = Chessboard()

    with open("1.in", "r") as f:
        lines = f.readlines()
        lines.pop(0)

        for line in lines:
            line = line.strip()
            if line != "-1 -1 -1 -1 -1 -1" and line != "":
                chessboard.act(line)

        in2 = ""
        for y in range(8):
            for x in range(8):
                if chessboard.board[y, x] == 1 or chessboard.board[y, x] == 2:
                    if chessboard.board[y, x] == chessboard.player:
                        in2 += "1 "
                    else:
                        in2 += "2 "
                else:
                    in2 += str(chessboard.board[y, x]) + " "
            
            in2 += "\n"
        
        with open("2.in", "w") as f:
            f.write(in2)
    
    print("1:")
    os.system("answer1.exe < 1.in")
    print("2:")
    os.system("answer2.exe < 2.in")

