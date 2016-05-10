import numpy as np

from random import choice

INF = 1e9


'''The online sparsified GPTD module.
'''
class GPTDModule(object):

    def __init__(self, initialState, initialAction):
        self.c_state = 10
        self.sigma_state = 0.2
        self.b_action = 0.1
        self.gamma = 0.9
        self.sigma = 1.
        self.eps = 0.1

        self.dictionary = [(initialState, initialAction)]
        # a is coefficient vector of projection onto dictionary
        self.a = np.ones(1)
        init = (initialState, initialAction)
        # K_tilde is kernel matrix of dictionary
        self.K_tilde_inv = np.array([[1. / self.fullKernel(init, init)]])
        self.alpha_tilde = np.zeros(1)
        self.C_tilde = np.zeros((1, 1))
        self.c_tilde = np.zeros(1)
        self.d = 0.
        self.s = INF
        self.nu = 0.1
        
        N = np.array((-1, 0))
        W = np.array((0, 1))
        S = np.array((1, 0))
        E = np.array((0, -1))

        self.actions = [N, W, S, E]

    
    def stateKernel(self, s1, s2): # state is r, c in the maze
        dr = s1[0] - s2[0]
        dc = s1[1] - s2[1]
        return self.c_state * np.exp(-(dr**2 + dc**2)/(2*self.sigma_state**2))

    def actionKernel(self, a1, a2): # action is 0, 1, 2, 3
        diff = abs(a1 - a2)
        if diff == 0:
            return 1
        elif diff == 2:
            dot = -1
        else:
            dot = 0
        return 1 + (1 - self.b_action)/(2*(dot - 1))

    def fullKernel(self, x1, x2):
        return self.stateKernel(x1[0], x2[0]) * self.actionKernel(x1[1], x2[1])

    def getKernelVec(self, x):
        k = np.zeros(len(self.dictionary))
        for i, xi in enumerate(self.dictionary):
            k[i] = self.fullKernel(xi, x)
        return k
            
    def getMaxAction(self, state):
        if len(self.dictionary) <= 1 or np.random.random() < self.eps:
            return choice(range(len(self.actions)))
        k_state = np.zeros(len(self.dictionary))
        for i, xi in enumerate(self.dictionary):
            k_state[i] = self.stateKernel(xi[0], state)
        print 'alphatilde', self.alpha_tilde
        beta = k_state * self.alpha_tilde
        u = np.zeros(2)
        for i, xi in enumerate(self.dictionary):
            u += beta[i] * self.actions[xi[1]]
        a = np.arctan2(u[1], u[0])
        veca = np.array([np.cos(a), np.sin(a)])

        # get the action closest to the optimal angle vector
        out = 0
        dot = 0
        for i, act in enumerate(self.actions):
            bdot = np.dot(act, veca)
            if bdot > dot:
                out = i
                dot = bdot
        return out


    def update(self, state, action, reward, newstate, newaction):
        # project new state onto the current dictionary
        oldx = (state, action)
        newx = (newstate, newaction)

        k_tilde = self.getKernelVec(newx)
        a_prev = self.a
        self.a = self.K_tilde_inv.dot(k_tilde)
        delta = self.fullKernel(newx, newx) - k_tilde.dot(self.a)
        k_tilde_prev = self.getKernelVec(oldx)
        delta_k_tilde = k_tilde_prev - k_tilde * self.gamma
        lambd = self.gamma * self.sigma**2 / self.s

        self.d = self.d * lambd + reward - np.dot(delta_k_tilde, self.alpha_tilde)

        print delta, 'delta'
        if (delta > self.nu): # adding to the dictionary if error is large
            r, c = self.K_tilde_inv.shape
            K_tilde_inv_t = np.zeros((r+1, c+1))
            K_tilde_inv_t[0:-1, 0:-1] = self.K_tilde_inv * delta + self.a * self.a.T
            K_tilde_inv_t[0:-1, -1] = -self.a
            K_tilde_inv_t[-1, 0:-1] = -self.a.T 
            K_tilde_inv_t[-1, -1] = 1
            K_tilde_inv_t /= delta
            self.K_tilde_inv = K_tilde_inv_t

            self.a = np.zeros(len(self.a) + 1)
            self.a[-1] = 1
            print 'a', self.a
            h_tilde = np.zeros(len(a_prev) + 1)
            h_tilde[:-1] = a_prev
            h_tilde[-1] = -self.gamma
            delta_k = a_prev.dot(k_tilde_prev - k_tilde * 2 * self.gamma) + \
                    self.gamma**2*self.fullKernel(newx, newx)
            self.s = (1+self.gamma**2) * self.sigma**2 + delta_k - \
                    delta_k_tilde.T.dot(self.C_tilde).dot(delta_k_tilde) - \
                    delta_k_tilde.T.dot(self.C_tilde).dot(delta_k_tilde) + \
                    2 * lambd * self.c_tilde.T.dot(delta_k_tilde) - \
                    lambd * self.gamma * self.sigma**2

            temp1 = np.zeros(len(self.c_tilde) + 1)
            temp2 = np.zeros(len(self.c_tilde) + 1)
            temp1[0:-1] = self.c_tilde
            temp2[0:-1] = self.C_tilde * delta_k_tilde
            self.c_tilde = temp1 * lambd + h_tilde - temp2

            temp = np.zeros(len(self.alpha_tilde) + 1)
            temp[:-1] = self.alpha_tilde
            self.alpha_tilde = temp

            r,c = self.C_tilde.shape
            self.C_tilde = np.vstack((np.hstack((self.C_tilde, np.zeros((r,1)))),
                np.zeros((1,c+1))))

            self.dictionary.append((newstate, newaction))

            print self.c_tilde
            if self.c_tilde[0] >= INF:
                print "NaN detected"

        else:
            h_tilde = a_prev - self.a * self.gamma
            delta_k = np.dot(h_tilde, delta_k_tilde)
            c_tilde_prev = self.c_tilde

            self.c_tilde = self.c_tilde * lambd + h_tilde - \
                    self.C_tilde * delta_k_tilde

            if self.c_tilde[0] >= INF:
                print "Positive Infinity detected."

            self.s = (1 + self.gamma**2) * self.sigma**2 + \
                    np.dot(delta_k_tilde, self.c_tilde + c_tilde_prev * lambd)- \
                    lambd * self.gamma * self.sigma**2

        self.alpha_tilde = self.alpha_tilde + self.c_tilde / float(self.s) * self.d

        self.C_tilde = self.C_tilde + self.c_tilde * self.c_tilde.T / self.s
