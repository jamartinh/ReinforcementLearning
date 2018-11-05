from math import pi

def int2bin(num,size=None):
    num =int(num)
    strbin = ""
    if num==0:
        strbin+='0'
    while num>0:
        strbin+= str( num % 2 )
        num = num / 2
    if not size:
        return strbin
    if len(strbin)<size:
        diff = size-len(strbin)
        strbin+='0'*(diff)
        
    return strbin


def bin2int(binlist):
    ret_val = 0
    for i in range(len(binlist)):
        if binlist[i]!='0':
            ret_val = ret_val + 2**i
    return ret_val



def GetBinaryCode(value,ranges,max_deep,max_error=0):
        a,b = ranges
        deep = 0
        binary_code =""
        error = 0.0
        while deep < max_deep:
            c = (a+b)/2.0
            if value>= c:
                binary_code+='1'
                a = c
            else:
                binary_code+='0'
                b = c
            deep = deep + 1
            error = abs(((a+b)/2.0)-value)
            if error<max_error:
                break
            
        return binary_code,error

def GetNumFromBianryCode(strbin,ranges):
    a,b     = ranges
    number  = a
    for bit in strbin:
        c = (a+b)/2.0        
        if bool(int(bit)):            
            a = c
        else:           
            b = c
    c =(((a+b)/2.0) + c)/2.0
    return c
                
            
                
            
rangos=[-10,10]
numero = 0
deep   = 5
print('Rangos: ',rangos)
print('Numero: ',numero)
print()
s,error =GetBinaryCode(numero,rangos,deep,max_error=0)
print('binary code',s,' error mark: ',error)
print()
print('retro conversion')
x = GetNumFromBianryCode(s,rangos)
print('numero',x)

print('error ',numero-x)
