import numpy as np

class Experiment(object):
    """ An experiment matches up a task with an agent and handles their interactions.
    """

    def __init__(self, task, agent):
        self.task = task
        self.agent = agent
        self.stepid = 0

    def doInteractions(self, number=1):
        """ The default implementation directly maps the methods of the agent and the task.
            Returns the number of interactions done.
        """
        for _ in range(number):
            self._oneInteraction()
        return self.stepid

    def _oneInteraction(self):
        """ Give the observation to the agent, takes its resulting action and returns
            it to the task. Then gives the reward to the agent again and returns it.
        """
        self.stepid += 1
        # observations from tasks are vectors
        self.agent.integrateObservation(self.task.getObservation())
        self.task.performAction(self.agent.getAction())
        reward = self.task.getReward()
        newstate = self.task.getObservation()
        self.agent.getReward(reward)
        return reward


if __name__=="__main__":

    from environments.loop import Loop, LoopTask
    from environments.chain import Chain, ChainTask
    from environments.maze import FlagMaze, FlagMazeTask
    from module import ActionModule
    from agent import BayesAgent

    # env = Loop()
    # task = LoopTask(env)

    # env = Chain()
    # task = ChainTask(env)

    # struct = np.array([[0, 0, 0, 0, 0],
    #                  [0, 1, 1, 0, 0],
    #                  [0, 1, 1, 1, 0],
    #                  [0, 1, 1, 1, 0],
    #                  [0, 1, 0, 1, 0],
    #                  [0, 0, 0, 0, 0]])
    # flagPos = [(3, 1)]
    # goal = (3, 3)

    struct = np.array([[0, 0, 0, 0, 0, 0, 0, 0, 0],
                       [0, 1, 1, 0, 1, 1, 1, 1, 0],
                       [0, 1, 1, 0, 1, 1, 0, 1, 0],
                       [0, 1, 1, 0, 1, 1, 0, 1, 0],
                       [0, 1, 1, 0, 1, 0, 0, 1, 0],
                       [0, 1, 1, 1, 1, 1, 0, 1, 0],
                       [0, 0, 0, 0, 0, 0, 0, 1, 0],
                       [0, 1, 1, 1, 1, 1, 1, 1, 0],
                       [0, 0, 0, 0, 0, 0, 0, 0, 0]])

    flagPos = [(5,1), (7, 3), (3, 5)]
    goal = (1, 7)

    env = FlagMaze(struct, flagPos, goal)
    task = FlagMazeTask(env)

    module = ActionModule(env.outdim, env.indim)
    agent = BayesAgent(module)
    exp = Experiment(task, agent)

    reward = 0
    xs = []
    ys = []

    import matplotlib.pyplot as plt

    for i in xrange(5000):
        exp.doInteractions(1)
        reward += agent.lastreward

        if i%50 == 0:
            xs.append(i)
            ys.append(reward)
        if agent.lastreward > 0:
            print "ACTION:",agent.lastaction, "STATE:",agent.laststate, "REWARD:",agent.lastreward
        # print env.curPos

    print "TOTAL REWARD:", reward
    print ys
    plt.plot(xs, ys)
    plt.show()

