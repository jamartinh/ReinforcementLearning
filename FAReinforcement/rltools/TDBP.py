from numpy import *
from numpy.random import *
from numpy.linalg import *
import psyco
psyco.full()




class TDBP():

    n=0           # number of inputs
    num_hidden=0  # number of hidden layer ?
    m= 0          # number of outputs
    MAX_UNITS = 1000 # maximum total number of units (to set array sizes)
    time_steps= 0 # number of time steps to simulate
    BIAS=1        # strength of the bias (constant input) contribution
    ALPHA=0.5     #;  /* 1st layer learning rate (typically 1/n) */
    BETA= 0.5     #;   /* 2nd layer learning rate (typically 1/num_hidden) */
    GAMMA = 1     #;  /* discount-rate parameter (typically 0.9) */
    LAMBDA = 0.9  #; /* trace decay parameter (should be <= gamma) */

    #/* Network Data Structure: */

    x = zeros((time_steps,n)) #; /* input data (units) */
    h = zeros(num_hidden)              # /* hidden layer */
    y = zeros(m)              # /* output layer */
    w = uniform(-2.,2,(num_hidden,m))  #; /* out weights */
    v = uniform(-2.,2,(n,num_hidden))  #; /* out weights */



    #/* Learning Data Structure: */

    old_y= zeros(m)
    ev   = zeros((n,num_hidden,m)) # /* hidden trace */
    ew   = zeros((num_hidden,m))           # /* output trace */
    r    = zeros((time_steps,MAX_UNITS))          # /* reward */
    error= zeros(MAX_UNITS)                       # /* TD error */
    t=0
                                              #;  /* current time step */

    def main(self):

        k=0;
        self.InitNetwork()

        t=0                   #/* No learning on time step 0 */
        self.Response()       #;            /* Just compute old response (old_y)...*/

        self.old_y = array(self.y)

        self.UpdateElig()          #/* ...and prepare the eligibilities */

        for t in range(1,self.time_steps+1): # /* a single pass through time series data */

           self.Response()         #/* forward pass - compute activities */
           for k in range(self.m):
               self.error[k] = self.r[t][k] + self.GAMMA*self.y[k] - self,old_y[k]      #; /* form errors */

           self.TDlearn()          #/* backward pass - learning */
           self.Response()         #/* forward pass must be done twice to form TD errors */
           self.old_y = array(self.y)
           self.UpdateElig()       #;       /* update eligibility traces */



    def InitNetwork(self):
        """
        * InitNetwork()
        *
        * Initialize weights and biases
        *
        """
        self.x[:,self.n]=self.BIAS
        self.h[self.num_hidden]=self.BIAS



    def Response(self,x_input):
        """/*****
        * Response()
        *
        * Compute hidden layer and output predictions
        *
        """
        self.x[0:self.n]=x_input
        self.h[num_hidden]=self.BIAS
        self.x[self.n]=self.BIAS

        for j in range(self.num_hidden):
            self.h[j]=0.0
            for i in range(self.n+1):
                self.h[j]+=self.x[i]*self.v[i,j]

            self.h[j]=tanh(self.h[j])

        for k in range(self.m):
            self.y[k]=0.0
            for j in range(self.num_hidden+1):
                self.y[k]+=self.h[j]*self.w[j,k]

            self.y[k]=tanh(self.y[k]) # tanh (OPTIONAL) */

        return self.y

    def TDlearn(td_error):
        """/*****
        * TDlearn()
        *
        *   Update weight vectors
        *
        """

        for k in range(self.m):
          for j in range (self.num_hidden+1):
              self.w[j,k]+=self.BETA*td_error[k]*self.ew[j,k]
              for i in range(self.n+1):
                  self.v[i,j]+=self.ALPHA*td_error[k]*self.ev[i,j,k]







    def UpdateElig(self):
        """/*****
        * UpdateElig()
        *
        * Calculate new weight eligibilities
        *
        """
        temp = 1.0-self.y**2

        for j in range(self.num_hidden+1):
            for k in range(self.m):
                self.ew[j,k]=self.LAMBDA*self.ew[j,k]+temp[k]*self.h[j]
                for i in range(self.n+1):
                    self.ev[i,j,k]=self.LAMBDA*self.ev[i,j,k]+temp[k]*self.w[j,k]*self.h[j]*(1.0-self.h[j])*self.x[i]

