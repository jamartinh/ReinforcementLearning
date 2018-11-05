from .SOMQ import SOMQ

Q = SOMQ(nactions=3,size_x=10,size_y=10,input_ranges=[[0,10],[0,10]],alpha=0.3)

#print Q.Net.W

s  =[1,1]
a  = 1

print('q(s,a)',Q(s,a))


for i in range(40):
    Q.Update(s,a,10)
    #print 'q(s,a)',Q.Net.W[:,:,2:2+3]
    print('Q(s,a)',Q(s,a))


s  =[5,5]
a  = 1
print('q(s,a)',Q(s,a))


for i in range(40):
    Q.Update(s,a,5)
    print('q(s,a)',Q(s,a))


s  =[10,10]
a  = 2
print('q(s,a)',Q(s,a))


for i in range(40):
    Q.Update(s,a,-5)
    print('q(s,a)',Q(s,a))


print()
print("===========================================================")
print("   TOTALS ")
print("===========================================================")
s  =[1,1]
a  = 1
print(s,a,'q(s,a)',Q(s,a))
print()
print()
s  =[5,5]
a  = 1
print(s,a,'q(s,a)',Q(s,a))
print()
print()
s  =[10,10]
a  = 2
print(s,a,'q(s,a)',Q(s,a))


