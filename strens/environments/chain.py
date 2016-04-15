from random import random

from pybrain.rl.environments.environment import Environment
from pybrain.rl.environments.task import Task


class Chain(Environment):
    '''A five state chain problem with two possible actions at every step.'''

    # number of action values
    indim = 2

    # number of sensor values
    outdim = 5

    state = 0
    probSlip = 0.2

    def getSensors(self):
        return [self.state,]

    def performAction(self, action):
        # action can be 0 or 1
        # 0 -> b, 1 -> a (with prob = 1 - probSlip)
        if random() < self.probSlip:
            # print "slipped!====================================================================="
            action = not action
        if action == 0:
            self.state = 0
        else:
            self.state = min(self.state + 1, 4)

    def reset(self):
        pass


class ChainTask(Task):

    def getReward(self):
        if self.env.state == 0:
            return 2
        if self.env.state == 4:
            return 10
        return 0

    def performAction(self, action):
        Task.performAction(self, int(action[0]))

    def getObservation(self):
        return self.env.getSensors()


if __name__=="__main__":
    # testing the environment and task

    from pybrain.rl.learners.valuebased import ActionValueTable
    from pybrain.rl.learners import Q
    from pybrain.rl.agents import LearningAgent
    from pybrain.rl.experiments import Experiment
    from pybrain.rl.explorers import EpsilonGreedyExplorer

    env = Chain()
    controller = ActionValueTable(env.outdim, env.indim)
    controller.initialize(1.)
#    controller.initialize(0.)

#    learner = Q(0.5, 0.8) # alpha 0.5, gamma 0.8
    learner = Q() # default alpha 0.5, gamma 0.99
#    learner._setExplorer(EpsilonGreedyExplorer(0.5))
    agent = LearningAgent(controller, learner)

    task = ChainTask(env)
    exp = Experiment(task, agent)

    reward = 0
    xs = []
    ys = []

    import matplotlib.pyplot as plt
    for i in xrange(5000):
        exp.doInteractions(1)
        agent.learn()

        reward += agent.lastreward

        if i%100 == 0:
            xs.append(i)
            ys.append(reward)
            print i
        # print learner.laststate, learner.lastaction, learner.lastreward
#        print controller.params.reshape(5, 2)

    print "TOTAL REWARD:", reward
    print ys
