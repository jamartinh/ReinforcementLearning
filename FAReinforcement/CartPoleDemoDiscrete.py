
from rltools.FARLBasic import *

#from Environments.CartPoleEnvironment import CartPoleEnvironment
from Environments.CartPoleEnvironmentG import CartPoleEnvironment

from rltools.lwprQ import lwprQ
from rltools.kNNSCIPY import kNNQ
from rltools.RNeuroQ import RNeuroQ
from rltools.NeuroQ import NeuroQ
from rltools.ActionSelection import *
import pickle
#from pylab import *


def CartPoleExperiment(Episodes=100,nk=0):

    print()
    print('===================================================================')
    print('           INIT EXPERIMENT','k='+str(nk+1))

    # results of the experiment
    x = list(range(1,Episodes+1))
    y =[]
    yr =[]

    #Build the Environment
    Env = CartPoleEnvironment()

    # Build a function approximator
    #Q = kNNQ(nactions=Env.nactions,input_ranges=Env.input_ranges,nelemns=[2,3,10,2],npoints=False,k=1,alpha=0.25)
    Q = kNNQ(nactions=Env.nactions,input_ranges=Env.input_ranges,nelemns=[2+7,3+7,10+3,2+7],npoints=False,k=4,alpha=0.3,lm=0.95)
    #Q = kNNQ(nactions=Env.nactions,input_ranges=Env.input_ranges,nelemns=False,npoints=100,k=4,alpha=0.3,lm=0.95)
    #Q = lwprQ(nactions=Env.nactions,input_ranges=Env.input_ranges)
    #Q = RNeuroQ(Env.nactions, Env.input_ranges, 200, Env.reward_ranges,alpha=0.3)
    #Q =  NeuroQ(Env.nactions, Env.input_ranges, 100, Env.reward_ranges,Env.deep_in,Env.deep_out,alpha=0.3)
    # Get the Action Selector
    As = e_greedy_selection(epsilon=0.0)
    #As = e_softmax_selection(epsilon=0.3)

    #Build the Agent
    CP = FARLBase(Q,Env,As,gamma=1.0)
    CP.Environment.graphs=True


    for i in range(Episodes):
        result = CP.SARSAEpisode(1000)
        #result = CP.QLearningEpisode(1000)
        CP.SelectAction.epsilon = CP.SelectAction.epsilon * 0.9
        CP.PlotLearningCurve(i,result[1],CP.SelectAction.epsilon)
        print('Episode;', str(i),'Total Reward:',str(result[0]),'Steps:',str(result[1]))
        y.append(result[1])
        yr.append(result[0])
##        if i==50:
##            miny =min(y)
##            figure(i)
##            plot(range(1,len(y)+1),y,'k')
##            title(r'$ k = 4, \quad  \lambda=0.95,\quad \epsilon=0 , \quad  \alpha=0.3 $')
##            grid('on')
##            axis([1, i, 0, 1100])
##            xlabel('Episodes')
##            ylabel('Steps')
##            savefig('cpresultdiscrete.pdf')
##            print "salvado"
##            close(i)



    CP.LearningCurveGraph.display.visible = False

    return [[x,y,nk],[x,yr,nk]]

def Experiments():
    results1=[]
    results2=[]
    for i in range(0,10):
        x = CartPoleExperiment(Episodes=200,nk=i)
        results1.append( x[0] )
        results2.append( x[1] )

    pickle.dump(results1,open('discretecartpolestepscq.dat','w'))
    pickle.dump(results2,open('discretecartpolerewardcq.dat','w'))



if __name__ == '__main__':
    #Experiments()
    x = CartPoleExperiment(100,3)
    pickle.dump(x[0],open('discretecartpolesteps.dat','w'))

