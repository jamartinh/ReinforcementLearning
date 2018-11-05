
from rltools.FARLBasic import *
from Environments.ProleEnvironment2 import ProleEnvironment
from rltools.kNNSCIPY import kNNQ
from rltools.lwprQ import lwprQ
#from rltools.RNeuroQ import RNeuroQ
from rltools.ActionSelection import *
import pickle
import time
#from pylab import *





def Experiment(Episodes=100,nk=1):
    print()
    print('===================================================================')
    print('           INIT EXPERIMENT','k='+str(nk+1))

    strfilename = "PDATA.npy"

    # results of the experiment
    x = list(range(1,Episodes+1))
    y =[]

    #Build the Environment
    Env = ProleEnvironment()

    # Build a function approximator
    #Q = kNNQ(nactions=Env.nactions,input_ranges=Env.input_ranges,nelemns=[8,8,8,8,8,8],npoints=False,k=1,alpha=0.3,lm=0.95)
    Q = kNNQ(nactions=Env.nactions,input_ranges=Env.input_ranges,nelemns=[8,8,8,8,8,8],npoints=False,k=2**6,alpha=10.5,lm=0.95)
    Q.Load(strfilename)

    #Experimental
    #Q = lwprQ(nactions=Env.nactions,input_ranges=Env.input_ranges)


    # Get the Action Selector
    As = e_greedy_selection(epsilon=0.1)
    #As = e_softmax_selection(epsilon=0.1)

    #Build the Agent
    RL = FARLBase(Q,Env,As,gamma=1.0)
    RL.Environment.graphs=True



    for i in range(Episodes):
        t1= time.clock()
        result = RL.SARSAEpisode(1000)
        #result = RL.QLearningEpisode(1000)
        t2 = time.clock()-t1
        RL.SelectAction.epsilon *= 0.9
        #RL.Q.alpha *= 0.995

        #RL.PlotLearningCurve(i,result[1],RL.SelectAction.epsilon)
        print('Episode',i,' Steps:',result[1],'Reward:',result[0],'time',t2,'alpha',RL.Q.alpha)
        #Q.Save(strfilename)
        y.append(result[1])


    return [x,y,nk]



def Experiments():
    results=[]
    for i in range(0,10):
        x = Experiment(Episodes=1000,nk=i)
        results.append( x )

    pickle.dump(results,open('proleresults.dat','w'))




if __name__ == '__main__':
    #Experiment(Episodes=10000,nk=3)
    Experiments()
