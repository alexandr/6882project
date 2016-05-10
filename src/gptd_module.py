import numpy as np

from random import choice

INF = 1e9


class GPTDModule(object):

    def __init__(self, initialState, initialAction):
        self.history = []
        self.c_state = 10
        self.sigma_state = 0.2
        self.b_action = 0.1
        self.alpha_tilde = np.zeros(1)
        self.gamma = 0.9
        self.sigma = 1.
        self.s = INF
        self.a = np.ones(1)
        self.K_tilde_inv = np.array([1. / fullKernel(initialState, initialAction)])
        self.C_tilde = np.zeros((1, 1))
        self.c_tilde = np.zeros(1)
        self.d = 0.
        self.nu = 0.1
        
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


    def update(state, action, reward, newstate, newaction):
        k_tilde = self.getKernelVec((newstate, newaction))
        a_prev = self.a
        self.a = self.K_tilde_inv * k_tilde
        delta = self.fullKernel((newstate, newaction)) - np.dot(k_tilde, self.a)
        k_tilde_prev = self.getKernelVec((state, action))
        delta_k_tilde = k_tilde_prev - k_tilde * self.gamma
        lambd = self.gamma * self.sigma * self.sigma / self.s

        self.d = self.d * lambd + reward - np.dot(delta_k_tilde, self.alpha_tilde)

        if (delta > self.nu):
            r, c = K_tilde_inv.shape
            K_tilde_inv_t = np.zeros((r+1, c+1))
            K_tilde_inv_t[0:-1, 0:-1] = self.K_tilde_inv * delta + self.a * self.a.T
            K_tilde_inv_t[0:-1, -1] = -self.a
            K_tilde_inv_t[-1, 0:-1] = -self.a.T 
            K_tilde_inv_t[-1, -1] = 1
            K_tilde_inv_t /= delta
            self.K_tilde_inv = K_tilde_inv_t

            self.a = np.zeros(len(self.a) + 1)
            a[-1] = 1
            h_tilde = np.zeros(len(self.a))
            h_tilde[0:-2] = a_prev
            h_tilde[-1] = -self.gamma
            delta_k = a_prev.dot(k_tilde_prev - k_tilde * 2 * self.gamma) + self.gamma**2*self.fullKernel(newstate, newaction)
            self.s = (1+self.gamma**2) * self.sigma**2 + delta_k - delta_k_tilde.T * self.C_tilde * delta_k_tilde \
                - delta_k_tilde.T * self.C_tilde * delta_k_tilde + 2 * lambd * np.dot(self.c_tilde, delta_k_tilde) - lambd * self.gamma * self.sigma * self.sigma

            temp1 = np.zeros(len(self.c_tilde) + 1)
            temp2 = np.zeros(len(self.c_tilde) + 1)
            temp1[0:-1] = self.c_tilde
            temp2[0:-1] = self.C_tilde * delta_k_tilde
            self.c_tilde = temp1 * lambd + h_tilde - temp2

            temp = np.zeros(len(self.alpha_tilde) + 1)
            temp[0:-1] = self.alpha_tilde
            self.alpha_tilde = temp

            r,c = self.C_tilde.shape
            temp = np.zeros((r+1, c+1))
            temp[0:-1, 0:-1] = self.C_tilde
            self.C_tilde = temp

            self.history.append((newstate, newaction))

            if self.c_tilde[0] >= INF:
                print "NaN detected"

        else:
            h_tilde = a_prev - self.a * self.gamma
            delta_k = np.dot(h_tilde, delta_k_tilde)
            c_tilde_prev = self.c_tilde

            self.c_tilde = self.c_tilde * lambd + h_tilde - self.C_tilde * delta_k_tilde

            if self.c_tilde[0] >= INF:
                print "Positive Infinity detected."

            self.s = (1 + self.gamma * self.gamma) * self.sigma * self.sigma + np.dot(delta_k_tilde, self.c_tilde + c_tilde_prev * lambd) - lambd * self.gamma * self.sigma * self.sigma

        self.alpha_tilde = self.alpha_tilde + self.c_tilde / self.s * self.d
        self.C_tilde = self.C_tilde + self.c_tilde * self.c_tilde.T / self.s













