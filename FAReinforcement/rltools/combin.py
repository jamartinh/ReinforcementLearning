



a = []
b = [4,5]
c = [7,8]

print(a)
print(b)
print(c)

def crossproduct(a,b):
    lista =[]

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
            lista.append(x1)            
            
    return lista
    
    
    
            
x1 = crossproduct(b,a)
print('result1:',x1)
print()
print()
x2 = crossproduct(x1,c)
print('result2:',x2)
