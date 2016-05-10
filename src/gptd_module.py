import numpy as np

from random import choice


class GPTDModule(object):

    def __init__(self):
        self.history = []
        self.c_state = 10
        self.sigma_state = 0.2
        self.b_action = 0.1
        self.alpha_tilde = None
        
        N = np.array((-1, 0))
        W = np.array((0, 1))
        S = np.array((1, 0))
        E = np.array((0, -1))

        self.actions = [N, W, S, E]

    
    def stateKernel(s1, s2): # state is r, c in the maze
        dr = s1[0] - s2[0]
        dc = s1[1] - s2[1]
        return self.c_state * np.exp(-(dr**2 + dc**2)/(2*self.sigma_state**2))

    def actionKernel(a1, a2): # action is 0, 1, 2, 3
        diff = abs(a1 - a2)
        if diff == 0:
            return 1
        elif diff == 2:
            dot = -1
        else:
            dot = 0
        return 1 + (1 - self.b_action)/(2*(dot - 1))

    def fullKernel(x1, x2):
        return stateKernel(x1[0], x2[0]) * actionKernel(x1[1], x2[1])

    def getKernelVec(self, x):
        k = np.zeros(len(self.history))
        for i, xi in enumerate(self.history):
            k[i] = fullKernel(xi, x)
            
    def getBestAction(self, state):
        if len(self.history) <= 1 or np.random.random() < self.eps:
            return choice(range(len(self.actions)))
        k_state = np.zeros(len(self.history))
        for i, xi in enumerate(self.history):
            k_state[i] = stateKernel(xi[0], state)
        beta = k_state * self.alpha_tilde
        u = np.zeros(2)
        for i, xi in enumerate(self.history):
            u += beta[i] * self.actions[xi[1]]
        a = np.arctan2(u[1], u[0])
        out = 0
        dot = 0
        for i, act in enumerate(self.actions):
            bdot = np.dot(act, a)
            if bdot > dot:
                out = i
                bdot = dot
        return out


    def update(state, action, reward, newstate):
        pass
