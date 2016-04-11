import numpy as np


class DirichletPrior(object):

    def __init__(self, numStates, alphas):
        assert len(alphas) == numStates,
        "Length of alpha input vector does not match number of states."
        self.numStates = numStates
        self.alphas = alphas


class SparsePrior(object):

    def __init__(self, numStates, zMin):
        self.numStates = numStates
        self.zMin = zMin


if __name__=="__main__":

    numStates = 3
    alphas = np.ones(numStates) * 1.0/numStates
    dp = DirichletPrior(3, alphas)
