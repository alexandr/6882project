import numpy as np
from scipy.special import gamma


class DirichletPrior(object):

    def __init__(self, alphas):
        self.alphas = alphas

    def likelihood(self, pis):
        '''pis are the multinomial probabilities'''
        unnormalized = np.product(np.power(pis, self.alphas-1))
        beta = np.product(gamma(self.alphas)) / gamma(np.sum(self.alphas))
        return unnormalized / beta


class SparsePrior(object):

    def __init__(self, numStates, zMin):
        self.numStates = numStates
        self.zMin = zMin


if __name__=="__main__":

    numStates = 3
    alphas = np.ones(numStates) * 1.0/numStates
    dp = DirichletPrior(3, alphas)
    pis = np.ones(numStates) * 1.0/numStates
    print dp.likelihood(pis)
