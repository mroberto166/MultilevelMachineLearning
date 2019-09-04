import numpy as np
import matplotlib.pyplot as plt

def f1(c,L):
    return r*1/(1-2**c)
def f2(c,L):
    return r*(L+1)
def f3(c,L):
    return r*(2**(c*L))/(1 - 2**(-c))

def f(x,r,d,L):
    e = np.log2(1/r)/L
    c = e-d
    print(c)
    return (x+1)*r/(1 - 2**(c/x))
L=6
r=32/1024

x=np.linspace(1,L,1000)


plt.plot(x, f(x,r,1,L))

plt.show()