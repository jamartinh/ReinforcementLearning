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
        self.Q=[]
        for i in range(nactions):
            self.Q.append(LWPR(self.ninputs,1))
            self.Q[i].init_D = D*eye(self.ninputs)
            self.Q[i].init_alpha = alpha*ones([self.ninputs,self.ninputs])
            #self.Q[i].init_lambda =0.999
            #self.Q[i].final_lambda =0.99999
            #self.Q[i].tau_lambda = 0.9999
            self.Q[i].meta=True
            self.Q[i].diag_only = False
            #self.Q[i].norm_in = array([1.89,0.14])
        self.alpha = alpha


        self.lbounds = []
        self.ubounds = []
        for r in input_ranges:
            self.lbounds.append(r[0])
            self.ubounds.append(r[1])

        self.lbounds = array(self.lbounds)
        self.ubounds = array(self.ubounds)

    def RescaleInputs(self,s):
        #return array(s)
        return self.ScaleValue(array(s),self.lbounds,self.ubounds,-1.0,1.0)

    def ResetTraces(self):
        pass

    def ScaleValue(self,x,from_a,from_b,to_a,to_b):
        return (to_a) + (((x-from_a)/(from_b-from_a))*((to_b)-(to_a)))


    def GetValue(self,s,a=None):
        """ Return the Q value of state (s) for action (a)
        """
        state   = self.RescaleInputs(s)

        if a==None:
            values=[]
            for i in range(self.nactions):
                values.append(self.Q[i].predict(state))
            return values


        return self.Q[a].predict(state)



    def Update(self,s,a,v):
        """ update action value for action(a)

        """
        state   = self.RescaleInputs(s)

        self.Q[a].update(state,v)

    def UpdateAll(self,s,v):
        """ update action value for action(a)

        """
        raise NotImplementedError

    def HasPopulation(self):
        return False

    def Population(self):
        raise NotImplementedError

