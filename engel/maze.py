import numpy as np
from random import random
import matplotlib.pyplot as plt

from pybrain.rl.environments.environment import Environment
from pybrain.rl.environments.task import Task


class FlagMaze(object):
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

    def showMaze(self):
        img = np.copy(self.maze)
        img[self.goal] = 3
        plt.figure()
        plt.imshow(img, interpolation='None')
        plt.show()


class FlagMazeTask(object):
    
    def __init__(self, env):
        self.env = env

    def getReward(self):
        if self.env.curPos == self.env.goal:
            reward = self.env.flagsCollected
            self.env.reset()
            return reward
        return 0

    def performAction(self, action):
        Task.performAction(self, int(action))

    def getObservation(self):
        return self.env.curPos
    

if __name__ == "__main__":

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
