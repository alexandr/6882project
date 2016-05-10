from pybrain.rl.agents.agent import Agent


class BayesAgent(Agent):
    '''BayesAgent contains a module (action value table) that keeps
    track of our estimates for Q, T, R, and auxillary values. The
    agent interacts with the task through the experiment to receive
    observations and rewards.'''

    laststate = None
    lastaction = None
    lastreward = None

    def __init__(self, module):
        self.module = module

    def integrateObservation(self, obs):
        '''First, observe the current state.
        If this is not the first step, i.e. laststate, lastaction,
        and lastreward are not None, update the module with all four
        fields.'''

        if self.laststate != None:
            self.module.update(self.laststate, self.lastaction,
                               obs[0], self.lastreward)

        self.laststate = obs[0]
        self.lastaction = None
        self.lastreward = None

    def getAction(self):
        '''Second, get the best action from the module (action value
        table). Should happen after we observe the current state.'''

        # NOTE: does not modify AV table
        self.lastaction = self.module.getMaxAction(self.laststate)
        return [self.lastaction,]

    def getReward(self, reward):
        '''Last, receive the reward from the experiment.
        The new state observation comes in the next observation,
        where the module update will be performed.'''
        self.lastreward = reward
