from .FAInterface import FARL
from numpy import *
from numpy.random import *
from numpy.linalg import *
import time


class SN(FARL):

    def __init__(self,nactions,input_ranges,nelems=[],k=[],alpha=0.3,lm=0.90):


        self.ndims      = len(nelems)
        self.Qshape     = nelems + [nactions]
        self.nelems     = array(nelems)

        self.k          = array(k)
        self.nactions   = nactions
        self.Q          = zeros(self.Qshape)
        self.e          = zeros(self.Qshape)

        self.w          = []
        self.alpha      = alpha
        self.lm         = lm #good 0.95


        self.a=[]
        self.b=[]

        for l,u in input_ranges:
            self.a.append(l)
            self.b.append(u)


        self.a = array(self.a)
        self.b = array(self.b)
        self.d = ((self.b-self.a)/(self.nelems-1.0))







    def Load(self,strfilename):
        self.Q = load(strfilename)

    def Save(self,strfilename):
        save(strfilename,self.Q)

    def ResetTraces(self):
        self.e*= 0.0

    def GetIndices(self,s):

            p = around( (s-self.a)/self.d).astype(int)
            self.maxs = clip(p+self.k+1,1,self.nelems)
            self.mins = clip(p-self.k,0,self.maxs-1)
            M    = []
            for i in range(self.ndims):
                M.append(slice(self.mins[i],self.maxs[i]))
            return M


    def CalckQValues(self,M):
        self.N = prod(self.maxs-self.mins)
        #print 'N',self.N,prod(self.maxs-self.mins)

        Qvalues = average( self.Q[M].reshape((self.N,self.nactions)),0)
        return Qvalues

    def GetValue(self,s,a=None):
        """ Return the Q value of state (s) for action (a)
        """
        M = self.GetIndices(s)

        if a==None:
            return self.CalckQValues(M)

        return self.CalckQValues(M)[a]


    def Update(self,s,a,v,gamma=1.0):
        """ update action value for action(a)
        """

        M = self.GetIndices(s)


        #cumulating traces
        #self.e[M,a] = self.e[M,a] +  self.ac[M].flatten()

        #replacing traces
        if self.lm>0:
            self.e[M]     =  0.0
            self.e[M+[a]] =  1.0#self.ac

            TD_error    =  v - self.GetValue(s,a)
            self.Q +=  self.alpha * (TD_error) * self.e
            self.e*= self.lm
        else:
            TD_error    =  v - self.GetValue(s,a)
            self.Q[M+[a]] +=  self.alpha * (TD_error)




    def HasPopulation(self):
        return False

    def Population(self):
        return None





