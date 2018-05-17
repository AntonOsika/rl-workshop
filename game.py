import random 
import pandas as pd
import matplotlib.pyplot as plt

def play(agent1, agent2, player=0):

    pieces = 21

    agents = [agent1, agent2]

    while True:
        move = agents[player].act(pieces)
        move += 1
        pieces -= move
        
        if pieces <= 0:
            print('player {} lost'.format(player))
            return (player + 1) % 2

        player = (player + 1 ) % 2

def play_n(agent1, agent2, n, plot=True):
    winners = []
    for i in range(n):
        winner = play(agent1, agent2, i%2)
        if winner == 0:
            agent1.feedback(1)
            agent2.feedback(-1)
        else:
            agent1.feedback(-1)
            agent2.feedback(1)
        agent1.new_episode()
        agent2.new_episode()

        winners.append(winner)
        
    if plot:
        pd.Series(winners).rolling(10, 10).mean().plot()
        plt.ylim([0, 1])
        plt.grid(1)
        plt.show()

    return winners

