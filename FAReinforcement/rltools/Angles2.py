from Numeric import *
from perm import *


class AnglePartition:
       
    def __init__(self,nactionvars=1,numpuntos=4.0):

        self.lb =0.0
        self.ub =2*pi
        self.rango = self.ub-self.lb
        self.numpuntos = numpuntos+0.0
        self.step = self.rango/self.numpuntos
        self.angles=arange(self.lb,self.ub,self.step).tolist()
        self.actionlist,self.nactions = self.GetActionList(nactionvars)
        self.unitary = [self.unit_from_angles(i) for i in self.actionlist]
        
    def GetActionList(self,n):
        action=[]
        cont=0
        for i in xselections(self.angles,n):
            action.append(i);
            cont=cont+1
        return action,cont
        
    
    def unit_from_angles(self,ang):
        
        #x = [sin(a)*sin(b)*sin(c)*sin(d) 
        #     sin(a)*sin(b)*sin(c)*cos(d) 
        #     sin(a)*sin(b)*cos(c)        
        #     sin(a)*cos(b)               
        #     cos(a) ]                    


        N = len(ang)
        u = ones(N+1).astype(float)
      
        for i in range(N):               
            for j in range(N-i):
                u[i] = u[i] * sin(ang[j])         

            if i>0:
                u[i] = u[i] * cos(ang[(N-1)-(i-1)])
               
       
        u[N]= cos(ang[0])
            
            
        return u.tolist()
            
            

if __name__ == '__main__':
    from visual import *
    x = AnglePartition(2,5)
    p = array(x.unitary)
    for i in range(x.nactions):
        sphere(pos=p[i],radius=0.05)
        #print p[i],sqrt(sum(p[0]))
        print(x.actionlist[i])
    while(1):
        rate(100)
  


