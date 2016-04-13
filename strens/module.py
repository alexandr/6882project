import numpy as np
from random import choice

from pybrain.rl.learners.valuebased.interface import ActionValueInterface

from priors import DirichletPrior, SparsePrior


'''
Updates to Q are given by
Q(s, a) = E[R(s, a)] + gamma * sum_s'[T(s, a, s') * max_a' Q(s', a')]

where we infer the transition probabilities
T(s, a, s') = p(s' | s, a)

and use the ML estimate for E[R(s, a)] = average of rewards received
from (s, a)
'''

class ActionModule(ActionValueInterface):
    '''The module that keeps track of the Q(s, a) estimates
    as well as the posterior distribution on T(s, a, s') and
    the ML estimates of E[R(s, a)]'''

    def __init__(self, numStates, numActions, alphas=None):
        self.numRows = numStates
        self.numColumns = numActions
        self.actionTable = np.zeros((numStates, numActions))

        # TODO: set actual alphas
        if alphas is None:
            alphas = np.ones(numStates)
        self.alphas = alphas
        self.prior = DirichletPrior(self.alphas)

        self.transitionProbs = np.zeros((numStates, numActions, numStates))
        self.initTransProbs()

        # The quantities we need to maintain in accordance with the Strens paper
        self.visitCount = np.zeros((numStates, numActions))
        self.sumRewards = np.zeros((numStates, numActions))
        self.sumSqRewards = np.zeros((numStates, numActions))
        self.successorStates = [[set() for _ in xrange(numActions)] for _ in xrange(numStates)]
        self.transitionCount = np.zeros((numStates, numActions, numStates))

        # ML estimates, updated using prioritized sweeping
        self.expectedReward = np.zeros((numStates, numActions))
        self.discountedReturn = np.zeros((numStates, numActions))

    def update(self, state, action, newstate, reward):
        self.visitCount[state][action] += 1
        self.sumRewards[state][action] += reward
        self.sumSqRewards[state][action] += reward * reward
        self.successorStates[state][action].add(newstate)
        self.transitionCount[state][action][newstate] += 1

        # TODO: update ML estimates using prioritized sweeping
        # TODO: update Qs according to formula

    @property
    def numActions(self):
        return self.numColumns

    @property
    def numStates(self):
        return self.numRows

    def initTransProbs(self):
        for s in xrange(self.numStates):
            for a in xrange(self.numActions):
                self.transitionProbs[s,a,:] = np.random.dirichlet(self.alphas)

    def getMaxAction(self, state):
        possible = self.actionTable[state]
        best = np.where(possible == max(possible))[0]
        return choice(best)
