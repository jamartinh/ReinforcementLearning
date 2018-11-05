from .FAInterface import FARL
from .rangeXbpnn import NN
from Numeric import *
from RandomArray import *
class  kRNeuroQ(FARL):
    
    
    def __init__(self,nactions, input_ranges, nhidden, output_ranges,alpha=0.3,k=50):
        self.nactions = nactions
        self.alpha = alpha
        self.k = k
        
        #Create the function approximator
        self.Net=[]
        for i in range(self.nactions):
            self.Net.append(NN(input_ranges, nhidden, output_ranges))  # the function approximator


    
    
    def GetValue(self,s,a=None):
        """ Return the Q value of state (s) for action (a)

        """
##        s = array(s)
##        # calculate k neighbor points
##        inputs = normal(0,0.2,(self.k+1,s.shape[0])) + s
##        inputs[self.k+1] = s
##
##        #calculate k neighbor points probabilities
##        d  = sum((inputs-s)**2,1)
##        ac = 1.0/(1.0+d)
##        ac /= sum(ac)
        
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
        s = array(s)
        
        # calculate k neighbor points
        inputs = normal(0,0.3,(self.k+1,s.shape[0])) + s
        inputs[self.k] = s
        
        #calculate k neighbor points probabilities
        #d  = sum((inputs-s)**2,1)
        #ac = 1.0/(1.0+d)
        #ac /= sum(ac)
        
        
        targets    = v + normal(0,0.3,(self.k+1))
        targets[self.k] = v
        pattern    = [[inputs,targets]]
        self.Net[a].array_train(inputs, targets, 1, 0.5, 0.1)

    def UpdateAll(self,s,v):
        """ update action value for action(a)

        """
        for i in range(self.nactions):
            self.Update(s,i,v[i])
            
    def HasPopulation(self):
        return False

            

