from __future__ import print_function
import random
from collections import defaultdict

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


class Game:
    def __init__(self, player1, player2):
        self.players = [player1, player2]
        self._reset()

    def _reset(self):
        self.pieces = 21
        self.active_player = random.randint(0, 1)

    def play(self, render=True, learn=True):
        self._reset()
        while self.pieces > 0:
            move = self.players[self.active_player].act(self.pieces)
            move += 1

            if render:
                print("Player {} takes {} down to {}".format(
                    self.active_player + 1, move, self.pieces - move))

            self.pieces -= move
            if self.pieces <= 0:
                print("Player {} lost".format(self.active_player + 1))

                if learn:
                    self.players[self.active_player].feedback(-1)
                
                self.active_player = (self.active_player + 1) % 2

                if learn:
                    self.players[self.active_player].feedback(1)

                return self.active_player

            self.active_player = (self.active_player + 1) % 2


def play(p1, p2, render=False, learn=True, seed=None):
    players = [p1, p2]
    pieces = 21
    active_player = random.randint(0, 1)
    if seed is not None:
        active_player = seed

    while pieces > 0:
        move = players[active_player].act(pieces)
        move += 1

        if render:
            print("Player {} takes {} down to {}".format(
                active_player + 1, move, pieces - move))

        pieces -= move
        if pieces <= 0:
            print("Player {} lost".format(active_player + 1))

            if learn:
                players[active_player].feedback(-1)
            
            active_player = (active_player + 1) % 2

            if learn:
                players[active_player].feedback(1)

            return active_player

        active_player = (active_player + 1) % 2


def play_n(p1, p2, n, learn=True):
    winners = []
    for i in range(n):
        winners.append(play(p1, p2, seed=i%2, learn=learn))
    print(np.mean(winners))
    print(np.mean(winners[:n//2]))
    print(np.mean(winners[n//2:]))
    pd.rolling_mean(pd.Series(winners), max(1, n//100), min_periods=1).plot()
    pd.Series(winners).rolling(len(winners), 1).mean().plot()
    plt.grid(1)
    plt.show()

