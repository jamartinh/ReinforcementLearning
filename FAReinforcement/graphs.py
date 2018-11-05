import pickle
from numpy import *

xc = pickle.load(open("contiuouscartpolesteps.dat","r"))
xd = pickle.load(open("discretecartpolesteps.dat","r"))
del xc[2]
del xd[2]
print(transpose(xc))
print(transpose(xd))


f=open('cartpoletxt.txt', 'w')
for i in range(len(xc[0])):
    a = xc[0][i]
    b = xd[1][i]
    c = xc[1][i]
    strline = str(a) + "\t" + str(b) + "\t" + str(c) +"\n"
    print(strline)
    f.write(strline)

f.close()

