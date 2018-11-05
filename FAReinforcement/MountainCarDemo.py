
from rltools.FARLBasic import *
from Environments.MountainCarEnvironment import MCEnvironment
#from rltools.kNNQ import kNNQ
from rltools.kNNSCIPY import kNNQ
from rltools.lwprQ import lwprQ
from rltools.NeuroQ import NeuroQ
from rltools.SOMQ import SOMQ
from rltools.RNeuroQ import RNeuroQ
from rltools.SN import SN

from rltools.ActionSelection import *
import pickle
import time
#from pylab import *



def MountainCarExperiment(Episodes=100,nk=1):
    print()
    print('===================================================================')
    print('           INIT EXPERIMENT','k='+str(nk+1))

    # results of the experiment
    x = list(range(1,Episodes+1))
    y =[]

    #Build the Environment
    MCEnv = MCEnvironment()

    # Build a function approximator
    #Q = kNNQ(nactions=MCEnv.nactions,input_ranges=MCEnv.input_ranges,nelemns=[10,5],npoints=False,k=nk+1,alpha=0.5)

    #best
    Q = kNNQ(nactions=MCEnv.nactions,input_ranges=MCEnv.input_ranges,nelemns=[10+10,5+10],npoints=False,k=nk+1,alpha=0.9,lm=0.95)
    #Q = kNNQ(nactions=MCEnv.nactions,input_ranges=MCEnv.input_ranges,nelemns=False,npoints=100,k=3**2,alpha=3,lm=0.0)
    #Q = SN(nactions=MCEnv.nactions,input_ranges=MCEnv.input_ranges,nelems=[20,15],k=[2,2],alpha=0.3,lm=0.0)
    #Experimental but acceptable performance
    #Q = lwprQ(nactions=MCEnv.nactions,input_ranges=MCEnv.input_ranges)
    #Q = NeuroQ(MCEnv.nactions, MCEnv.input_ranges, 60, MCEnv.reward_ranges,MCEnv.deep_in,MCEnv.deep_out,alpha=0.3)



    # Experimental and not functional at this momment.
    #Q = DkNNQ(nactions=MCEnv.nactions,input_ranges=MCEnv.input_ranges,nelemns=[10+10,5+10],npoints=False,k=nk+1,alpha=0.5)
    #Q = RNeuroQ(MCEnv.nactions, MCEnv.input_ranges, 10, MCEnv.reward_ranges,alpha=0.3)
    #Q = kRNeuroQ(MCEnv.nactions, MCEnv.input_ranges, 10, MCEnv.reward_ranges,alpha=0.3)
    #Q = SOMQ(nactions=MCEnv.nactions,size_x=10,size_y=10,input_ranges=MCEnv.input_ranges,alpha=0.3)

    # Get the Action Selector
    As = e_greedy_selection(epsilon=0.001)
    #As = e_softmax_selection(epsilon=0.1)

    #Build the Agent
    MC = FARLBase(Q,MCEnv,As,gamma=1.0)
    MC.Environment.graphs=True
    MC.Environment.PlotPopulation(MC.Q)




    for i in range(Episodes):
        t1= time.clock()
        result = MC.SARSAEpisode(1000)
        #MC.Q.PushtTaces()
        MC.Q.ResetTraces()
        #result = MC.QLearningEpisode(1000)
        t2 = time.clock()-t1
        MC.SelectAction.epsilon *= 0.9
        #MC.Q.alpha *= 0.995

        MC.PlotLearningCurve(i,result[1],MC.SelectAction.epsilon)
        MC.Environment.PlotPopulation(MC.Q)
        print('Episode',i,' Steps:',result[1],'time',t2,'alpha',MC.Q.alpha)

        y.append(result[1])
##        if i==50:
##
##            figure(i)
##            plot(range(1,len(y)+1),y,'k')
##            title(r'$ \min=105,\quad k = 4, \quad  \lambda=0.95, \quad  \epsilon=0.0, \quad \alpha=0.9 $')
##            grid('on')
##            axis([1, i, 0, 800])
##            xlabel('Episodes')
##            ylabel('Steps')
##            savefig('mcresult.pdf')
##            print "salvado"
##            close(i)


    return [x,y,nk]



def Experiments():
    results=[]
    for i in range(0,10):
        x = MountainCarExperiment(Episodes=200,nk=i)
        results.append( x )

    pickle.dump(results,open('mcresultsknnql.dat','w'))




if __name__ == '__main__':
    MountainCarExperiment(Episodes=1000,nk=3)
    #Experiments()
