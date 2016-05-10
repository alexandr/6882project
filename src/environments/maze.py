import numpy as np
from random import random
import matplotlib.pyplot as plt

from pybrain.rl.environments.environment import Environment
from pybrain.rl.environments.task import Task


class FlagMaze(Environment):
    '''A maze with flags to be collected.
    The agent starts at the start state, collects flags on
    the way to the goal, and is immediately transported back
    to the start state upon reaching the goal state.
    Maximize the flags collected.'''

    maze = None
    states = None
    goal = None
    start = (1, 1)
    flagsCollected = 0

    # directions
    N = (-1, 0)
    W = (0, 1)
    S = (1, 0)
    E = (0, -1)

    directions = [N, W, S, E]
    
    # number of possible actions
    indim = len(directions)

    def __init__(self, maze, flagPos, goal):
        # 0 means wall, 1 means open, 2 means flag
        self.maze = maze
        r, c = maze.shape
        self.numRows = r
        self.numColumns = c

        self.states = np.copy(maze)
        self.states[maze == 0] = -1
        self.states[maze > 0] = np.arange(len(maze[maze > 0]))
        
        # number of possible sensor values
        self.outdim = r * c

        self.goal = goal
        self.flags = flagPos
        self.reset()

    def reset(self):
        self.curPos = self.start
        self.flagsCollected = 0
        for flag in self.flags:
            self.maze[flag] = 2

    def _moveInDir(self, pos, d):
        '''returns the position in the direction d from p'''
        return (pos[0] + d[0], pos[1] + d[-1])

    def _canMove(self, pos):
        ''' returns whether we can move to this pos'''
        if self.maze[pos] == 0:
            return False
        return True

    def performAction(self, action):
        d = self.directions[action]
        newpos = self._moveInDir(self.curPos, d)
        if self._canMove(newpos):
            if self.maze[newpos] == 2:
                self.flagsCollected += 1
                self.maze[newpos] = 1
            self.curPos = newpos

    def getSensors(self):
        '''returns the state of the four surrounding squares'''
        obs = np.ones(len(FlagMaze.directions))
        for i, d in enumerate(FlagMaze.directions):
            newpos = self._moveInDir(self.curPos, d)
            if self._canMove(newpos):
                obs[i] = self.maze[newpos]
            else:
                obs[i] = 0
        return obs

    def showMaze(self):
        img = np.copy(self.maze)
        img[self.goal] = 3
        plt.figure()
        plt.imshow(img, interpolation='None')
        plt.show()


class FlagMazeTask(Task):

    def getReward(self):
        if self.env.curPos == self.env.goal:
            reward = self.env.flagsCollected
            self.env.reset()
            return reward
        return 0

    def performAction(self, action):
        Task.performAction(self, int(action[0]))

    def getObservation(self):
        return [self.env.states[self.env.curPos],]
#        x, y = self.env.curPos
#        return [x * self.env.numColumns + y,]
    

if __name__ == "__main__":

    from pybrain.rl.learners.valuebased import ActionValueTable
    from pybrain.rl.learners import Q
    from pybrain.rl.agents import LearningAgent
    from pybrain.rl.experiments import Experiment
    from pybrain.rl.explorers import EpsilonGreedyExplorer

    easy = np.array([[0, 0, 0, 0, 0],
                     [0, 1, 1, 0, 0],
                     [0, 1, 1, 1, 0],
                     [0, 1, 1, 1, 0],
                     [0, 1, 0, 1, 0],
                     [0, 0, 0, 0, 0]])
    easyFlags = [(3, 1)]
    easyGoal = (3, 3)

    hard = np.array([[0, 0, 0, 0, 0, 0, 0, 0, 0],
                     [0, 1, 0, 1, 1, 0, 1, 1, 0],
                     [0, 1, 0, 1, 1, 0, 1, 1, 0],
                     [0, 1, 1, 1, 1, 1, 1, 1, 0],
                     [0, 1, 1, 1, 1, 1, 1, 1, 0],
                     [0, 0, 0, 1, 1, 1, 0, 0, 0],
                     [0, 1, 1, 1, 0, 1, 1, 0, 0],
                     [0, 1, 1, 1, 0, 1, 1, 0, 0],
                     [0, 0, 0, 0, 0, 0, 0, 0, 0]])

#    hardFlags = [(1,3), (7, 2), (5, 6)]
    hardFlags = [(7,3), (3, 5)]
    hardGoal = (1, 7)

    env = FlagMaze(hard, hardFlags, hardGoal)
    controller = ActionValueTable(env.outdim, env.indim)
    controller.initialize(1.)
#    controller.initialize(0.)

#    learner = Q(0.5, 0.8) # alpha 0.5, gamma 0.8
    learner = Q() # default alpha 0.5, gamma 0.99
#    learner._setExplorer(EpsilonGreedyExplorer(0.5))
    agent = LearningAgent(controller, learner)

    task = FlagMazeTask(env)
    exp = Experiment(task, agent)

    import matplotlib.pyplot as plt

    reward = 0
    xs = []
    ys = []

    for i in xrange(5000):
        exp.doInteractions(1)
        agent.learn()
        reward += agent.lastreward

        if i%50 == 0:
            xs.append(i)
            ys.append(reward)
            print i
        #print learner.laststate, learner.lastaction, learner.lastreward

#        print controller.params.reshape(5, 2)
    print "TOTAL REWARD:", reward
    print ys
    plt.plot(xs, ys)
    plt.show()
