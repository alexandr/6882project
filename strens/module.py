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

    def __init__(self, numStates, numActions, alphas=None, gamma=0.99):
        self.numRows = numStates
        self.numColumns = numActions

        # Q table
        # HACK: setting these all to 1's to allow for exploration
        self.actionTable = np.ones((numStates, numActions))

        # I have no idea how to set this
        self.gamma = gamma

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

        self.rewardGammaAlpha = 10.
        self.rewardGammaBeta = 1.

        self.muNot = 10.
        self.nNot = 1.


        self.sumSqRewards = np.zeros((numStates, numActions))
        self.successorStates = [[set() for _ in xrange(numActions)] for _ in xrange(numStates)]
        self.transitionCount = np.zeros((numStates, numActions, numStates))

        # ones for our prior lol
        # HACK: try different priors
        self.transitionDirichletParams = np.ones((numStates, numActions, numStates))

        # ML estimates, updated using prioritized sweeping
        self.expectedReward = np.zeros((numStates, numActions))
        self.discountedReturn = np.zeros((numStates, numActions))

    def update(self, state, action, newstate, reward):
        self.visitCount[state][action] += 1
        self.sumRewards[state][action] += reward
        self.sumSqRewards[state][action] += reward * reward
        self.successorStates[state][action].add(newstate)
        self.transitionCount[state][action][newstate] += 1

        # update transition probability params
        self.transitionDirichletParams[state][action][newstate] += 1

        # update reward expectation parameters
        
        # update Qs according to formula
        self.updateAllQValues()

        # print "TRANSITION PROBABILITIES FROM:", state
        # print self.transitionDirichletParams[state, :, :]


        # print "Q TABLE from:", state
        # print self.actionTable[state]
        

    def updateAllQValues(self):
        newQvalues = np.zeros((self.numStates, self.numActions))

        for s in xrange(self.numStates):
            for a in xrange(self.numActions):
                newQvalues[s][a] = self.getUpdatedQValue(s, a)

        self.actionTable[:] = newQvalues

    def getUpdatedQValue(self, state, action):
        transitionProb = np.random.dirichlet(self.transitionDirichletParams[state, action, :])
        def sumArg(otherState):
            return transitionProb[otherState] * self.actionTable[otherState][self.getMaxAction(otherState)]


        # calculate appropriate alpha and beta for the Gamma

        n = float(self.visitCount[state, action])
        alpha = self.rewardGammaAlpha + 0.5 * n

        meanX = float(self.sumRewards[state, action]) / n if n > 0 else 0.
        nNot = self.nNot
        muNot = self.muNot

        part1 = 0.5 * n * meanX * meanX - meanX * self.sumRewards[state, action] + 0.5 * self.sumSqRewards[state, action]
        part2 = 0.5 * n * nNot * (meanX - muNot) * (meanX - muNot) / (n + nNot)

        # print part1, part2

        beta = self.rewardGammaBeta + part1 + part2

        # print "ALPHA:", alpha, "BETA:", beta

        tau = np.random.gamma(alpha, 1. / beta)

        muNot = n / (n + self.nNot) * meanX + self.nNot / (n + self.nNot) * self.muNot
        sigmasqNot = n * tau + self.nNot * tau

        mu = np.random.normal(muNot, 1./np.sqrt(sigmasqNot))

        expectedStateAction = np.random.normal(mu, 1./np.sqrt(tau))
        # print "XMEAN:", meanX
        # print "MU:", mu, "SIGMA:", 1/np.sqrt(tau)

        return expectedStateAction + self.gamma * sum(sumArg(s) for s in xrange(self.numStates))
        # print "STATE:", state, "ACTION:", action, "UPDATED EXP REWARD:", self.actionTable[state][action]
        # print "TRANSITION PROBABILITIES FROM:", state
        # for a in xrange(self.numActions):
        #     for s in xrange(self.numStates):
        #         print "newstate", s, "action", a, "probability:", float(self.transitionDirichletParams[state, a, s]) / normalizer
        # print self.transitionDirichletParams[state, :, :]

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
