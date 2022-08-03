import numpy as np
import math
from math import exp
from math import sqrt
from random import uniform 
from random import seed
import matplotlib.pyplot as plt
#variables
sw2 = 17
spx2 = 2.6
CpW = 58.1
k2 = 0.4
by = 0.0035
By = 1.2e-06
V_3 = 15.5
a2 = 150
V_2 = 44.2
b2 = 0.00499
B2 = 1.35e-06
Sl = 1.7
fl = 9
ex1 = 2.3
sw1 = 15
spx1 = 1.9
b1 = 0.0011
B1 = 1.79904862716992e-06
De = 57.2617535188108
a = 3.1
V_1 = 0.196

Ec=418.39#energy conversion factor

ET=60*Ec#The total energy of the system in a.m.u*Angstrom/(Picosecond)^2
mu=16.8# linear reduced mass 

def PES(x,y):
    q2=180*y/3.141
    V1=-b1*q2**2+B1*q2**4+De*(1-exp(-a*(x-1.5)))**2+V_1
    S1=1/(1+exp(-sw1*(x-spx1)))
    V2=-a2*(x-2.1)**2+V_2-b2*q2**2+B2*q2**4+Sl*exp(fl*(x-ex1))
    V3=-CpW*exp((-k2/(x-1)**4)*q2**2)-by*q2**2 +By*q2**4+V_3
    S2=1/(1+exp(-sw2*(x-spx2)))
    v=V1*(1-S1)+S1*V2*(1-S2)+S2*V3
    return v*Ec#in a.m.u*Angstrom/(Picosecond)^2

#function initializing variables
def Initialize(Xrange=[-15,15],Yrange=[-10,10],TotE=120): 
    x,y=uniform(Xrange[0],Xrange[1]),uniform(Yrange[0],Yrange[1])
    V=PES(x,y)
    KE=(TotE-V)#in a.m.u*Angstrom/(Picosecond)^2
    KEx=uniform(0,KE)
    KEy=KE-KEx
    
    sign=uniform(-1,1)
    sign=sign/abs(sign)
    vx=sign*sqrt(KEx*2/mu)
    
    sign=uniform(-1,1)
    sign=sign/abs(sign)
    my = mu*x**2
    vy=sign*sqrt(KEy*2/my)


    return x,y,vx,vy,V

def F(x,y): #return accelarations; inputs: x(in angstroms), y(in radians); outputs: Ax(Kcal/Angstrom),Ay(Kcal/rad)  
    #Energy Functions
    q2=y*180/3.141#radian to degree conversion
    V1=-b1*q2**2+B1*q2**4+De*(1-exp(-a*(x-1.5)))**2+V_1
    S1=1/(1+exp(-sw1*(x-spx1)))
    V2=-a2*(x-2.1)**2+V_2-b2*q2**2+B2*q2**4+Sl*exp(fl*(x-ex1))
    V3=-CpW*exp((-k2/(x-1)**4)*q2**2)-by*q2**2 +By*q2**4+V_3
    S2=1/(1+exp(-sw2*(x-spx2)))
    #Derivative functions
    dS1 = sw1*S1*(1-S1)
    dS2 = sw2*S2*(1-S2)

    dV1_dx = 2*De*a*(1-exp(-a*(x-1.5)))*exp(-a*(x-1.5))
    dV1_dy = -2*b1*q2+4*B1*q2**3

    dV2_dx = -2*a2*(x-2.1)+Sl*fl*exp(fl*(x-ex1))
    dV2_dy = -2*b2*q2+4*B2*q2**3

    dV3_dx = (-4*k2*q2**2*CpW/(x-1)**5)*exp((-k2/(x-1)**4)*q2**2)#derivatives(kcal/A)
    dV3_dy = (2*k2*q2*CpW/(x-1)**4)*exp((-k2/(x-1)**4)*q2**2) - 2*by*q2 + 4*By*q2**3#(Kcal/degre)

    Fx=-(dV1_dx*(1-S1) - V1*dS1 + dS1*V2*(1-S2) - dS2*S1*V2 + S1*(1-S2)*dV2_dx + dS2*V3 + S2*dV3_dx)#Force in x direction,unit is Kcal/degree
    Fy=-((1-S1)*dV1_dy + S1*(1-S2)*dV2_dy + S2*dV3_dy)#Torque in y direction
    
    #unit conversion is needed
    Fx=Fx*Ec#linear acceleration; unit:Angstrom/s^2
    Fy=Fy*180/(3.141)*Ec# angular acceleration; unit: 1/s^2. 180/3.141 multiplied of conversion back to radians
    
    return Fx,Fy 


def step_Euler(xi,yi,vxi,vyi,axi,ayi,dt): #basic Euler 
    #simple euler method for integration
    vxn = vxi + axi*dt
    vyn = vyi + ayi*dt

    xn = xi + (vxi + vxn)*dt/2 + 0.5*axi*dt**2
    yn = yi + (vyi + vyn)*dt/2 + 0.5*ayi*dt**2

    Fxn,Fyn = F(xn,yn)
    my = mu*xn**2
    axn = Fxn/mu + xn*vyn**2
    ayn = Fyn/my - 2*vxn*vyn/xn

    return xn, yn, vxn, vyn, axn, ayn 

def step_Euler_mod(xi,yi,vxi,vyi,axi,ayi,dt):#modified Euler, acombination of RK4 and Verlet 
    #step 1
    vx_half = vxi + axi*dt/2
    vy_half = vyi + ayi*dt/2

    x_half = xi + vxi*dt/2 + axi*dt**2/8
    y_half = yi + vyi*dt/2 + ayi*dt**2/8

    Fx_half,Fy_half = F(x_half,y_half)
    my = mu*x_half**2
    ax_half = Fx_half/mu + x_half*vy_half**2
    ay_half = Fy_half/my - 2*vx_half*vy_half/x_half
    
    #step 2
    vxn = vx_half + ax_half*dt/2 
    vyn = vy_half + ay_half*dt/2

    xn = x_half + vx_half*dt/2 + ax_half*dt**2/8
    yn = y_half + vy_half*dt/2 + ay_half*dt**2/8

    Fxn,Fyn = F(xn,yn)
    my = mu*xn**2
    axn = Fxn/mu + xn*vyn**2
    ayn = Fyn/my - 2*vxn*vyn/xn

    #Step 3[Final calculation]
    vxf = vxi + (axi + 2*ax_half + axn)*dt/4
    vyf = vyi + (ayi + 2*ay_half + ayn)*dt/4
    
    xf, yf, axf, ayf = xn, yn, axn, ayn


    return xf, yf, vxf, vyf, axf, ayf

def run_Euler(xinit,yinit,vxinit,vyinit,Steps=1000,dt=0.01):
    Fxinit,Fyinit = F(xinit,yinit)
    #total Moment of inertia I
    my = mu*xinit**2
    #
    vxt, vyt = vxinit, vyinit
    axt = Fxinit/mu + xinit*vyt**2
    ayt = Fyinit/my - 2*vxt*vyt/xinit
    xt, yt = [xinit], [yinit]
    Vx = [vxt]
    Vy = [vyt]
    e = PES(xt[-1],yt[-1]) + 0.5*mu*vxt**2 + 0.5*my*vyt**2
    E = [e]
    for i in range(Steps):
        x_new, y_new, vxt, vyt, axt, ayt = step_Euler_mod(xt[-1],yt[-1],vxt,vyt,axt,ayt,dt)
        xt.append(x_new)
        yt.append(y_new)
        myn = mu*x_new**2
        Vx.append(vxt)
        Vy.append(vyt)
        e = PES(x_new,y_new) + 0.5*mu*vxt**2 + 0.5*myn*vyt**2
        E.append(e)
    return xt,yt,Vx,Vy,E


from time import time
import os

dt=1e-4#time unit is picosecond
TotalTime=3e-1#time unit is picosecond
Steps=int(TotalTime/dt)


#testing Euler method

xinit,yinit,vxinit,vyinit,V_0 = Initialize([1.4,1.6],[-0.15,0.15],ET)
T=[dt*i for i in range(Steps+1)]

print("Euler Method")
X,Y,Vx,Vy,E=run_Euler(xinit,yinit,vxinit,vyinit,Steps,dt)

Xi=np.copy(X)
Yi=np.copy(Y)

T=[dt*i for i in range(Steps+1)]

print(f"Total Energy: {ET/Ec}Kcal/mol")
print(f"Totaltime of integration:{TotalTime}ps\nTime step:{dt}ps")
print(f"Relative error in E={(max(E)-min(E))/min(E)}[maximum deviation in Energy/Initial Total Energy]")

print(f"relative error in E={(max(E)-min(E))/min(E)}")
print(f"Initial Conditions:{xinit}\t{yinit}\t{vxinit}\t{vyinit}\t{V_0}")
print(f"Final Conditions:{X[-1]}\t{Y[-1]}\t{Vx[-1]}\t{Vy[-1]}\t{V_0}")

fig = plt.figure()
ax = plt.axes()
ax.plot(X,Y)


xl,yl,vxl,vyl=X[-1],Y[-1],Vx[-1],Vy[-1]
X,Y,Vx,Vy,E=run_Euler(xl,yl,-vxl,-vyl,Steps,dt)
print(f"Initial Conditions:{xl}\t{yl}\t{-vxl}\t{-vyl}")
print(f"Final Conditions:{X[-1]}\t{Y[-1]}\t{Vx[-1]}\t{Vy[-1]}")
print(f"Error in X,Y,Vx,Vy={xinit-X[-1]}\t{yinit-Y[-1]}\t{vxinit+Vx[-1]}\t{vyinit+Vy[-1]}")


plt.title("Euler")
ax.plot(X,Y,label="Back")
plt.legend()
plt.show()
file=f'Euler_energy_data.dat'
print(f"writing Energy to: {file}")
with open(file,'w') as f:
    for j in range(len(T)):
        f.write(f"{T[j]}\t{X[j]}\t{Y[j]}\t{E[j]}\n")
print("done")

'''print("Velocity_verlet")
X,Y,Px,Py,E=run(xinit,yinit,pxinit,pyinit,Steps,dt)

Xi=np.copy(X)
Yi=np.copy(Y)
T=[dt*i for i in range(Steps+1)]

print(f"relative error in E={(max(E)-min(E))/min(E)}")
print(f"Initial Conditions:{xinit}\t{yinit}\t{pxinit}\t{pyinit}\t{V_0}")
print(f"Final Conditions:{X[-1]}\t{Y[-1]}\t{Px[-1]}\t{Py[-1]}\t{V_0}")

xl,yl,pxl,pyl=X[-1],Y[-1],Px[-1],Py[-1]
X,Y,Px,Py,E=run(xl,yl,-pxl,-pyl,Steps,dt)
print(f"Initial Conditions:{xl}\t{yl}\t{-pxl}\t{-pyl}")
print(f"Final Conditions:{X[-1]}\t{Y[-1]}\t{Px[-1]}\t{Py[-1]}")
print(f"Error in X,Y,Vx,Vy={xinit-X[-1]}\t{yinit-Y[-1]}\t{pxinit+Px[-1]}\t{pyinit+Py[-1]}")

plt.plot(X,Y,label="Back")
plt.plot(Xi,Yi)
plt.title("Verlet")
plt.legend()
plt.show()'''