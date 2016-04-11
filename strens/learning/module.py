import numpy as np

from pybrain.structure.modules import Module
from pybrain.rl.learners.valuebased.interface import ActionValueInterface

from priors import DirichletPrior, SparsePrior


class ActionModule(Module, ActionValueInterfac):
    '''The module that keeps track of the Q(s, a) estimates
    as well as the posterior distribution on T(s, a, s') and
    the ML estimates of E[R(s, a)]'''

    def __init__(self, numStates, numActions):
        Module.__init__(self, 1, 1)

        self.numRows = numStates
        self.numColumns = numActions
        self.actionTable = np.zeros((numStates, numActions))
        self.transitionProbs = np.zeros((numStates, numActions, numStates))

    @property
    def numActions(self):
        return self.numColumns

    @property
    def numStates(self):
        return self.numRows
