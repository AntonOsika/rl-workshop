import game
import agents

# Game has 3 actions
mc = agents.MC(3)
p = agents.Perfect()

# Play Monte Carlo agent against perfect player:
game.play_n(mc, p, 1000)  # -> learns to win 50% of the time after about 500 games

mc.learn = False  # freeze agent
mc2 = agents.MC(3)

game.play_n(mc, mc2, 1000)  # -> new agent learns to beat first (frozen) agent

# Recreate MC and Q-learning agents:
mc = agents.MC(3)
q = agents.Q(3)

game.play_n(mc, q, 1000)  # -> q learning wins all the time, MC is stuck in local optima

# Recreate MC and Q-learning agents, and give MC 5% exploration
mc = agents.MC(3)
q = agents.Q(3)

mc.eps = 0.05

game.play_n(mc, q, 3000)  # -> they both learn to act optimally after ~700 games (but exporation makes MC lose sometimes)

# Remove exploration from MC and see if it is perfect
mc.eps = 0.0
game.play_n(mc, q, 1000)  # -> they win exactly 50% of games each as expected
