import numpy as np
from random import random

from pybrain.rl.environments.environment import Environment
from pybrain.rl.environments.task import Task


class FlagMaze(Environment):
    '''A maze with flags to be collected.
    The agent starts at the start state, collects flags on
    the way to the goal, and is immediately transported back
    to the start state upon reaching the goal state.
    Maximize the flags collected.'''

    maze = None
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
        
        # number of possible sensor values
        self.outdim = r * c

        self.goal = goal
        self.flags = flagPos
        self.reset()

    def reset(self):
        self.curPos = self.start
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


class FlagMazeTask(Task):

    def getReward(self):
        if self.env.curPos == self.env.goal:
            self.env.reset()
            return self.env.flagsCollected
        return 0

    def performAction(self, action):
        Task.performAction(self, int(action[0]))

    def getObservation(self):
        x, y = self.env.curPos
        return [x * self.env.numColumns + y,]
#        return self.env.getSensors()
    

if __name__ == "__main__":

    from pybrain.rl.learners.valuebased import ActionValueTable
    from pybrain.rl.learners import Q
    from pybrain.rl.agents import LearningAgent
    from pybrain.rl.experiments import Experiment
    from pybrain.rl.explorers import EpsilonGreedyExplorer

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
    controller = ActionValueTable(env.outdim, env.indim)
    controller.initialize(1.)
#    controller.initialize(0.)

#    learner = Q(0.5, 0.8) # alpha 0.5, gamma 0.8
    learner = Q() # default alpha 0.5, gamma 0.99
#    learner._setExplorer(EpsilonGreedyExplorer(0.5))
    agent = LearningAgent(controller, learner)

    task = FlagMazeTask(env)
    exp = Experiment(task, agent)

    for i in xrange(100):
        exp.doInteractions(1)
        agent.learn()
        print learner.laststate, learner.lastaction, learner.lastreward
#        print controller.params.reshape(5, 2)
