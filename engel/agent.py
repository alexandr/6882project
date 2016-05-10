'''Learning agent for learning with GPTD, using the
GPTDModule. Updates must be done with pairs (laststate, lastaction)
and (curstate, curaction), as specified in GPTDModule.
'''

class GPTDAgent(object):

    curstate = None
    nextaction = None
    laststate = None
    lastaction = None
    lastreward = None

    def __init__(self, module):
        self.module = module
    
    def integrateObservation(self, obs):
        # observation MUST be in form (r, c)
        self.curstate = obs

    def getAction(self):
        self.nextaction = self.module.getMaxAction(self.curstate)
        return self.nextaction


    def getReward(self, reward):    
        self.lastreward = reward
        # do the update
        self.module.update(self.laststate, self.lastaction,
                self.lastreward, self.curstate, self.nextaction)

        # move this iteration into the past
        self.laststate = self.curstate
        self.lastaction = self.nextaction
