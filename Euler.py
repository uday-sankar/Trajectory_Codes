# A trial code to test Euler method

def F(x): # F denotes the derivative
    f = 2*x # here the function is a parabola 
    return f
def step(yinit,xinit,dx):
    k1 = F(xinit)
    k2 = F(xinit + k1*dx/2)
    k3 = F(xinit + k2*dx/2)
    k4 = F(xinit + k3*dx)
    yn = yinit + (k1 + 2*k2 + 2*k3 + k4)*dx/6
    return yn

import numpy as np
dx=0.1
X = np.arange(-5,5,dx)
Y = [25] #initial value for y
for x in X[1:]:
    yn = step(Y[-1],x,dx)
    Y.append(yn)
import matplotlib.pyplot as plt
plt.plot(X,Y)
plt.show()
