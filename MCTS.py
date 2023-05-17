import random
import numpy
from model import NeuralNet
import game
from pprint import pprint


class MCTS:
    def __init__(self, net: NeuralNet, c_puct=5, simulate_times=100):
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
            self.Ps[s], v = self.net.predict(
                board, player, stage, preActionX, preActionY
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
                print("All valid moves were masked, doing a workaround.")
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
