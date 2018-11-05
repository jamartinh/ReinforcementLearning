from .FAInterface import FARL
from .rangeXbpnn import NN
from Numeric import *
from RandomArray import *
class  SNeuroQ(FARL):


    def __init__(self,nactions, input_ranges, nhidden, output_ranges,alpha=0.3):
        self.nactions = nactions
        self.alpha = alpha

        #Create the function approximator

        self.Net=NN(input_ranges, nhidden, output_ranges)  # the network

        self.traces    = []
        self.states    = []
        self.maxtraces = 100
        self.v0        = 9999999999999999
        self.steps     = 0




    def GetAction(self,s):
        """ Return the Q value of state (s) for action (a)

        """
        if random()>self.alpha:
            x =self.Net.update(s)
            return x[0]
        else:

            return uniform(-1,1)




    def Update(self,s,a,v):
        """ update action value for action(a)

        """

        dR = v-self.v0
        self.v0 = v
        inputs     = s
        targets    = a
        pattern    = [[inputs,targets]]

        if dR>0:
            if s not in self.states:
                self.traces.append([dR,pattern])
                self.states.append(s)
                self.Net.fast_train(pattern,1,0.5,0.1)


        targets    = -array(a)#uniform(-1,1)
        pattern    = [[inputs,targets]]
        self.Net.fast_train(pattern,1,0.5,0.1)


        self.traces.sort(reverse=True)
        if len(self.traces)>self.maxtraces:
            self.traces=self.traces[0:self.maxtraces]
            self.states=self.states[0:self.maxtraces]

        self.steps+=1
        if self.steps>=100:
            N = len(self.traces)
            for i in range(N):
                self.Net.fast_train(self.traces[i][1],1,0.5,0.1)
            self.steps=0
            #self.traces=[]






    def HasPopulation(self):
        return False



