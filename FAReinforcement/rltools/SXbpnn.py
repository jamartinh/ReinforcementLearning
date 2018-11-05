# Back-Propagation Neural Network for estimated value learning
# 
# Written in Python with Numeric python 
# 
# Jose Antonio Martin H <jamartin AT dia fi upm es>
from Numeric import *
from RandomArray import *
import pickle
#from numpy import * # use this if you use numpy instead of Numeric
import time

seed(3,3)


def GetBinaryCode(value,ranges,max_deep):
        a,b = ranges
        deep = 0
        binary_code =[]
        error = 0.0
        while deep < max_deep:
            c = (a+b)/2.0
            if value>= c:
                binary_code+=[1.0]
                a = c
            else:
                binary_code+=[-1.0]
                b = c
            deep = deep + 1
            
        return binary_code

def GetNumFromBianryCode(listbin,ranges):
    a,b     = ranges
    number  = a
    for bit in listbin:
        c = (a+b)/2.0        
        if bit>=0:            
            a = c
        else:           
            b = c
    c =(((a+b)/2.0) + c)/2.0
    return c


class NN:  

    def DecodeBinaryEncoding(self,values,ranges,deep):
        numbers_list=[]        
        for i in range(len(ranges)):
            numbers_list+= [GetNumFromBianryCode(values[i*deep:i*deep+deep],ranges[i])]
        return  numbers_list

    def CreateBinaryEncoding(self,values,ranges,deep):
        input_list=[]
        for i in range(len(ranges)):            
            input_list+= GetBinaryCode(values[i],ranges[i],deep) 

        return input_list
        
    
    def sigmoid(self,x):
        return tanh(x)

    # derivative of our sigmoid function
    def dsigmoid(self,y):
        return 1.0-y*y

    
    def __init__(self, input_ranges, nhidden, output_ranges,deep_in=10,deep_out=10):
        # number of input, hidden, and output nodes

        self.deep_in       = deep_in
        self.deep_out      = deep_out
        self.input_ranges  = input_ranges
        self.output_ranges = output_ranges
        
        self.ni = len(input_ranges)*deep_in + 1 # +1 for bias node
        self.nh = nhidden
        self.no = len(output_ranges)*deep_out

        # activations for nodes
        self.ai = ones((self.ni),Float)
        self.ah = ones((self.nh),Float)
        self.ao = ones((self.no),Float)
              
        
        # create weights
        self.wi = uniform(-2.0,2.0,(self.ni, self.nh))
        self.wo = uniform(-2.0,2.0,(self.nh, self.no))
        

        # last change in weights for momentum   
        self.ci = zeros((self.ni, self.nh),Float)
        self.co = zeros((self.nh, self.no),Float)


        # approx deltas
        self.output_deltas  = self.dsigmoid(self.ao) * (0*self.ao)
        self.hidden_deltas  = self.dsigmoid(self.ah) * matrixmultiply(self.wo,self.output_deltas) 

        # estimates learning rate
        self.est_lr = 0.5

    def SaveW(self,filename):
         W = [self.wi,self.wo]
         pickle.dump(W,open(filename,'w'))
         

    def LoadW(self,filename):         
         W = pickle.load(open(filename,'r'))
         self.wi=W[0]
         self.wo=W[1]

    def update(self, inputs):

        inputs = self.CreateBinaryEncoding(inputs,self.input_ranges,self.deep_in)
        
        if len(inputs) != self.ni-1: 
            raise ValueError('wrong number of inputs')

        # input activations
        self.ai[0:self.ni-1]=array(inputs)

       
        # hidden activations
        sum = matrixmultiply(transpose(self.wi),self.ai)
        self.ah = tanh(sum)
        
        # output activations
        sum = matrixmultiply(transpose(self.wo),self.ah)
        self.ao = tanh(sum)           

        self.out_values = self.DecodeBinaryEncoding(self.ao,self.output_ranges,self.deep_out)
        return self.out_values


    def backPropagate(self, targets_real, N, M):
        
        #targets_real = self.out_values  + 0.3* (array(targets_real) - self.out_values)
        
        targets_binary = self.CreateBinaryEncoding(targets_real,self.output_ranges,self.deep_out)

                
        if len(targets_binary) != self.no:
            raise ValueError('wrong number of target values')

        # calculate error terms for output
        od =  self.dsigmoid(self.ao) * (targets_binary-self.ao)
        self.output_deltas = od

        # calculate error terms for hidden
        error = matrixmultiply(self.wo,self.output_deltas)
        hd    =  self.dsigmoid(self.ah) * error
        self.hidden_deltas = hd
        
        # update output weights
        change = self.output_deltas * reshape(self.ah,(self.ah.shape[0],1))
        self.wo  = self.wo + N  * change + M * self.co        
        self.co = change
        

        # update input weights
        change = self.hidden_deltas * reshape(self.ai,(self.ai.shape[0],1))
        self.wi = self.wi + N*change + M*self.ci        
        self.ci = change 


        # calculate error        
        error = sum(0.5 * (targets_binary-self.ao)**2)
        #error = sum(0.5 * (self.out_values-array(targets_real))**2)
        return error


    def test(self, patterns):
        for p in patterns:
            print(p[0], '->', self.update(p[0]))


    def singletrain(self,inputs,targets_real):
        self.update(inputs)        
        error = self.backPropagate(targets_real,0.5, 0.1)        
        return error
         

        
    def train(self, patterns, iterations=100, N=0.5, M=0.1):
        # N: learning rate
        # M: momentum factor
        for i in range(iterations):
            error = 0.0
            for p in patterns:
                inputs = p[0]
                targets = p[1]
                self.update(inputs)
                error = error + self.backPropagate(targets, N, M)                
            if i % 100 == 0 and i!=0:
                print('error ' + str(error))


def demoavg():    

        
    # create a network with two input, two hidden, and one output nodes    
    rang_input  = [[1,20]]
    rang_output = [[1,20]]
    deep_in  = 5
    deep_out = 5
      
    
        
    a = time.clock()
    n = NN(rang_input, 5, rang_output ,deep_in,deep_out)

    targets = n.CreateBinaryEncoding([7],rang_output,deep_out)
    print('prueba targets ',targets)
    #return
    #train it with some patterns
    print("Starting single step training")
    for i in range(10000):
            error = 0.0
            # for 1
            inputs = [1]
            targets = [normal(5,0.5)] #noisy input
            error = error + n.singletrain(inputs,targets)
            
            # for 5
            inputs = [10]
            targets = [normal(8,0.5)] #noisy input
            error = error + n.singletrain(inputs,targets)
            
            # for 10
            inputs = [17]
            targets = [normal(15,0.5)] #noisy input
            error = error + n.singletrain(inputs,targets)
           
            if i % 100 == 0 and i!=0:
                print('error ' + str(error))
            #if error < 1:
            #        break
    # test it
   
    

    #n.test(pat)
    # for 1
    print("[1]===>",n.update([1]))
    print("[10]===>",n.update([10]))
    print("[17]===>",n.update([17]))
    
    b=time.clock()
    print("Total time for Back Propagation Trainning ",b-a)
    
   

    

if __name__ == '__main__':
    demoavg()




