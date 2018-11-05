from numpy import *
from math import radians

class AnglePartition:
       
    def __init__(self,nactionvars = 1,npartitions = 10):
        
        self.actionlist = self.CreateFullspace(nactionvars,npartitions)
        self.nactions   = len(self.actionlist)
        self.unitary    = [self.unit_from_angles(i) for i in self.actionlist]       
    
    
    def CreateFullspace(self,nvars,nangles):
        d=[]
        for i in range(nvars):
            r = [-pi , pi]
            n = nangles
            #x = linspace(r[0],r[1],n+1)
            xdiv = (r[1]-r[0])/float(n)
            x = arange(r[0],r[1]+xdiv,xdiv)
            d.append(list(x))

        space = d[0]
        for i in range(1,len(d)):
            space = self.crossproduct(space,d[i])

        return space
    
    def crossproduct(self,a,b):
        ret_list =[]

        for e1 in a:
            for e2 in b:
                if type(e1)!=type([]):
                    x1 = [e1]
                else:
                    x1 = list(e1)

                if type(e2)!=type([]):
                    x2 = [e2]
                else:
                    x2 = list(e2)

                x1+=x2
                ret_list.append(x1)

        return ret_list
        
    
    def unit_from_angles(self,ang):

        if type(ang)!=type([]):
            ang = [ang]
        
        N = len(ang)            
        u = ones(N+1)+0.00
      
        for i in range(N):               
            for j in range(N-i):
                u[i] = u[i] * cos(ang[j])
            if i>0:
                u[i] = u[i] * sin(ang[(N-1)-(i-1)])
               
       
        u[N] = sin(ang[0])
        return u.tolist()
            
            
if __name__ == '__main__':
    from visual import *
    x = AnglePartition(2,10)
    p = array(x.unitary)
    for i in range(x.nactions):
        sphere(pos=p[i],radius=0.05)
        #arrow(pos=p[i]*0,axis=p[i],shaftwidth = 0.01)
        #print p[i],sqrt(sum(p[0]))
        #print x.actionlist[i]
    while(1):
        rate(100)
  


