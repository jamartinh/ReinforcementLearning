from .FAInterface import *
from lwpr import *
from numpy import *
# interface that express the function approximator for
# Reinforcement Learning Algorithms.

class lwprQ(FARL):

    def __init__(self,nactions,input_ranges,D=50.000,alpha=250.0,lm=0.95):
        self.nactions = nactions
        self.input_ranges = input_ranges
        self.ninputs =len(input_ranges)

        self.Q            = LWPR(self.ninputs,self.nactions)
        self.Q.init_D     = D * eye(self.ninputs)
        self.Q.init_alpha = alpha * ones([self.ninputs,self.ninputs])
        self.Q.meta       = True
        self.Q.diag_only  = False
        #self.Q.penalty    = 0.9
        self.Q.w_gen      = 0.5

        self.alpha = alpha


        self.lbounds = []
        self.ubounds = []
        for r in input_ranges:
            self.lbounds.append(r[0])
            self.ubounds.append(r[1])

        self.lbounds = array(self.lbounds)
        self.ubounds = array(self.ubounds)

    def ResetTraces(self):
        pass

    def RescaleInputs(self,s):
        #return array(s)
        return self.ScaleValue(array(s),self.lbounds,self.ubounds,-1.0,1.0)

    def ScaleValue(self,x,from_a,from_b,to_a,to_b):
        return (to_a) + (((x-from_a)/(from_b-from_a))*((to_b)-(to_a)))

    def GetValue(self,s,a=None):
        """ Return the Q value of state (s) for action (a)
        """

        state   = self.RescaleInputs(s)
        if a==None:
            values = self.Q.predict(state)
            return values


        return self.Q.predict(state)[a]



    def Update(self,s,a,v):
        """ update action value for action(a)

        """
        state   = self.RescaleInputs(s)

        values = self.Q.predict(state)
        values[a]=v
        self.Q.update(state,values)

    def UpdateAll(self,s,v):
        """ update action value for action(a)

        """
        raise NotImplementedError

    def HasPopulation(self):
        return False

    def Population(self):
        raise NotImplementedError

