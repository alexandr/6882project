import numpy as np

from pybrain.rl.learners.valuebased.valuebased import ValueBasedLearner

'''Bayesian Q learning, 
Q(s, a) = E[R(s, a)] + gamma * sum_s'[T(s, a, s') * max_a' Q(s', a')]

where we infer the transition probabilities
T(s, a, s') = p(s' | s, a)

and use the ML estimate for E[R(s, a)] = average of the rewards received
from (s, a)
'''


class BayesianQ(ValueBasedLearner):

    # Q learning learns the optimal policy independently of the agent's actions
    offPolicy = True

    def __init__(self):
        ValueBasedLearner.__init__(self)

        self.laststate = None
        self.lastaction = None
