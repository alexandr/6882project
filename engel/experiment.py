import numpy as np

class GPTDExperiment(object):
    """ An experiment matches up a task with an agent and handles their interactions.
    """

    def __init__(self, task, agent):
        self.task = task
        self.agent = agent
        self.stepid = 0

    def doInteractions(self, iters=1):
        for _ in range(iters):
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

    from maze import FlagMaze, FlagMazeTask
    from module import GPTDModule
    from agent import GPTDAgent

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

    struct = np.array([[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                       [0, 1, 1, 0, 1, 1, 1, 1, 0, 1, 1, 1, 1, 0],
                       [0, 1, 1, 0, 1, 1, 0, 1, 1, 1, 1, 1, 0, 0],
                       [0, 1, 1, 1, 1, 1, 1, 1, 0, 1, 0, 1, 0, 0],
                       [0, 1, 1, 1, 1, 1, 0, 1, 0, 1, 0, 1, 1, 0],
                       [0, 1, 1, 0, 1, 0, 0, 1, 0, 1, 0, 1, 0, 0],
                       [0, 1, 1, 0, 1, 0, 0, 1, 0, 1, 0, 1, 0, 0],
                       [0, 1, 1, 0, 1, 1, 0, 1, 0, 1, 0, 1, 0, 0],
                       [0, 1, 0, 0, 1, 1, 0, 1, 1, 1, 1, 1, 1, 0],
                       [0, 1, 1, 1, 1, 1, 1, 1, 0, 0, 1, 0, 1, 0],
                       [0, 1, 0, 0, 1, 1, 0, 1, 0, 1, 1, 1, 1, 0],
                       [0, 1, 1, 0, 1, 1, 0, 1, 0, 1, 1, 1, 1, 0],
                       [0, 1, 1, 0, 1, 1, 0, 1, 0, 1, 1, 1, 1, 0],
                       [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]])

    flagPos = [(4,1), (4, 5), (2, 9), (1,11), (7, 7), (10, 1), (7, 9), (4, 3), (5,9), (5, 11), (11,11), (11,12), (12,12), (9,4), (9,5)]
    goal = (1, 12)

    env = FlagMaze(struct, flagPos, goal)
    task = FlagMazeTask(env)

    module = GPTDModule(env.start, 0)
    agent = GPTDAgent(module, env.start, 0)
    exp = GPTDExperiment(task, agent)

    reward = 0
    xs = []
    ys = []

    import matplotlib.pyplot as plt

    for i in xrange(0):
        print i
        exp.doInteractions(1)
        reward += agent.lastreward

        if i%50 == 0:
            xs.append(i)
            ys.append(reward)
        if agent.lastreward > 0:
            print "ACTION:",agent.lastaction, "STATE:",agent.laststate, "REWARD:",agent.lastreward
        print env.curPos

    print "TOTAL REWARD:", reward
    print ys
    plt.plot(xs, ys)
    plt.show()

    env.showMaze()

