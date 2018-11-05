from lwpr import *
from numpy import *
from random import *
from math import *

def cross_2D(x1,x2): 
   return max(exp(-10.0*x1*x1), exp(-50.0*x2*x2), 
               1.25*exp(-5.0*(x1*x1 + x2*x2)))

model = LWPR(2,2)

R = Random()
x = zeros(2)
y = zeros(2)

model.init_D = 50*eye(2)
model.init_alpha = 250*ones([2,2])

for i in range(1000):
   x[0] = R.uniform(-1,1)
   x[1] = R.uniform(-1,1)
   y[0] = cross_2D(x[0],x[1]) + R.gauss(0,0.1)
   y[1] = tanh(i)
   model.update(x,y)   
   
for x[0] in arange(-1,1,0.04):
   for x[1] in arange(-1,1,0.04):
      y = model.predict(x)
      #print "%6.2f %6.2f %8.4f"%(x[0],x[1],y[0])
      print(x,y,type(y))

print(model)
print(model.kernel)
print(model.norm_in)
#model.write_XML("cross2d.xml")
