from .FAInterface import FARL
#from rangeXbpnn import NN
from .Pbpnn import NN
class  RNeuroQ(FARL):


    def __init__(self,nactions, input_ranges, nhidden, output_ranges,alpha=0.3):
        self.nactions = nactions
        self.alpha = alpha

        #Create the function approximator
        self.Net=[]
        for i in range(self.nactions):
            self.Net.append(NN(input_ranges, nhidden, output_ranges))  # the function approximator

        self.mem=[]



    def GetValue(self,s,a=None):
        """ Return the Q value of state (s) for action (a)

        """
        if a==None:
            v = []
            for i in range(self.nactions):
                v.extend( self.Net[i].update(s))
            return v

        v = self.Net[a].update(s)
        if len(v)==1:
            return v[0]

        return v


    def Update(self,s,a,v):
        """ update action value for action(a)

        """
        inputs     = s
        targets    = [v]
        pattern    = [[inputs,targets]]
        self.Net[a].fast_train(pattern,3,self.alpha,0.1)
        #self.mem.append([s,a,v])
        #if len(self.mem)>30:
        #    self.ProcMem()

    def ProcMem(self):
        last_v = 0
        for xp in self.mem:
            inputs     = xp[0]
            targets    = 0.5*xp[2]+0.5*last_v#self.mem[0][2]
            pattern    = [[inputs,targets]]
            self.Net[xp[1]].fast_train(pattern,1,self.alpha,0.1)
            last_v = xp[2]
        self.mem=[]


    def UpdateAll(self,s,v):
        """ update action value for action(a)

        """
        for i in range(self.nactions):
            self.Update(s,i,v[i])

    def HasPopulation(self):
        return False



