from random import random

from pybrain.rl.environments.environment import Environment
from pybrain.rl.environments.task import Task


class Loop(Environment):
    '''A nine state loop problem with two possible actions at every step.'''
    # number of action values
    indim = 2

    # number of sensor values
    outdim = 9

    state = 0
    laststate = None
    lastaction = None

    def getSensors(self):
        return [self.state,]

    def performAction(self, action):
        # action can be 0 (a) or 1 (b)
        self.laststate = self.state
        self.lastaction = action
        if action == 0:
            self.state = min(self.state + 1, 5) % 5
        else:
            if self.state == 0:
                self.state = 5
            else:
                u = self.state / 5
                self.state = ((self.state + 1) % 5) + 5*u
                if self.state == 9:
                    self.state = 0

    def reset(self):
        pass


class LoopTask(Task):

    def getReward(self):
        if self.env.state == 0:
            if self.env.laststate == 4:
                return 1
            elif self.env.laststate == 8:
                return 2
            else:
                return 0
        return 0

    def performAction(self, action):
        Task.performAction(self, int(action[0]))

    def getObservations(self):
        return self.env.getSensors()


if __name__ == "__main__":

    from pybrain.rl.learners.valuebased import ActionValueTable
    from pybrain.rl.learners import Q
    from pybrain.rl.agents import LearningAgent
    from pybrain.rl.experiments import Experiment
    from pybrain.rl.explorers import EpsilonGreedyExplorer

    env = Loop()
    controller = ActionValueTable(env.outdim, env.indim)
    controller.initialize(1.)
#    controller.initialize(0.)

#    learner = Q(0.5, 0.8) # alpha 0.5, gamma 0.8
    learner = Q() # default alpha 0.5, gamma 0.99
#    learner._setExplorer(EpsilonGreedyExplorer(0.5))
    agent = LearningAgent(controller, learner)

    task = LoopTask(env)
    exp = Experiment(task, agent)

    import matplotlib.pyplot as plt

    reward = 0
    xs = []
    ys = []

    for i in xrange(5000):
        exp.doInteractions(1)
        agent.learn()
        reward += agent.lastreward
        print i

        if i%50 == 0:
            xs.append(i)
            ys.append(reward)
        print learner.laststate, learner.lastaction, learner.lastreward
#        print controller.params.reshape(5, 2)

    print "TOTAL REWARD:", reward
    print ys
    plt.plot(xs, ys)
    plt.show()
