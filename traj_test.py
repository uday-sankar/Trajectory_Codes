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

ET=40*Ec#The total energy of the system in a.m.u*Angstrom/(Picosecond)^2
mx=16.8# linear reduced mass 
my=16.8*1.4*1.4# the moment of inertia for the rotation; unit Kg*A^2

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
    px=sign*sqrt(KEx*2*mx)
    
    sign=uniform(-1,1)
    sign=sign/abs(sign)
    py=sign*sqrt(KEy*2*my)

    return x,y,px,py,V

def py(x,y,px,TotE=120):
    V=PES(x,y)
    KE=(TotE-V)
    s=uniform(-1,1)
    s=s/abs(s)
    py=s*sqrt(my*(2*KE-px**2/mx))
    return py

def InitializeP(x,y,TotE=ET): 
    V=PES(x,y)
    KE=(TotE-V)*418.4
    KEx=uniform(0,KE)
    KEy=KE-KEx
    
    sign=uniform(-1,1)
    sign=sign/abs(sign)
    px=sign*sqrt(KEx*2*mx)
    
    sign=uniform(-1,1)
    sign=sign/abs(sign)
    py=sign*sqrt(KEy*2*my)

    return px,py,V

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
    ax=Fx/mx*Ec#linear acceleration; unit:Angstrom/s^2
    ay=Fy*180/(3.141*my)*Ec# angular acceleration; unit: 1/s^2. 180/3.141 multiplied ofr conersion back to radians
    # Kcal/degree is converted to Kcal/rad in the nest line
    return ax,ay 

def step(xi,yi,vxi,vyi,axi=0,ayi=0,dt=0.01):#the inital cordinateds for each step Velocity verlet
    
    #axi,ayi=F(xi,yi)#initial accelerations(nth step)
    
    vx_half=vxi + 0.5*axi*dt#velocity ast half points n+1/2
    vy_half=vyi + 0.5*ayi*dt
    
    xn=xi + vx_half*dt# new displacements displacements at n+1 step
    yn=yi + vy_half*dt 
    
    axn,ayn=F(xn,yn) #new accelerations (at n+1 step) 
    
    vxn=vx_half + 0.5*axn*dt # new velocity (at n+1 step)
    vyn=vy_half + 0.5*ayn*dt 

    return xn, yn , vxn , vyn, axn, ayn

def run(xinit,yinit,pxinit,pyinit,Steps=1000,dt=0.01):
    #print("Initial Py:",pyinit)
    axt,ayt=F(xinit,yinit)
    xt,yt=[xinit],[yinit]
    vxt,vyt=pxinit/mx,pyinit/my
    Px=[pxinit]
    Py=[pyinit]
    e=PES(xt[-1],yt[-1]) + 0.5*mx*vxt**2 + 0.5*my*vyt**2
    E=[e]
    for i in range(Steps):
        x_new, y_new, vxt, vyt, axt, ayt=step(xt[-1],yt[-1],vxt,vyt,axt,ayt,dt)
        xt.append(x_new)
        yt.append(y_new)
        Px.append(vxt*mx)
        Py.append(vyt*my)
        e=PES(xt[-1],yt[-1]) + 0.5*mx*vxt**2 + 0.5*my*vyt**2
        E.append(e)
    return xt,yt,Px,Py,E

def run_multP(xinit,yinit,NumTraj=10,steps=1000,dt=0.01,TotE=101):
    Xt=[]
    Yt=[]
    Vxt=[]
    Vyt=[]
    Et=[]
    for i in range(NumTraj):
        px,py,v=InitializeP(xinit,yinit,TotE)
        xt,yt,vxt,vyt,E=run(xinit,yinit,px,py,steps,dt)
        Xt.append(xt)
        Yt.append(yt)
        Et.append(E)
        Vxt.append(vxt)
        Vyt.append(vyt)
    return Xt,Yt,Vxt,Vyt,Et

def run_Till(xinit,yinit,pxinit,pyinit,Xend=3,MaxSteps=1000,dt=0.01):
    #print("Initial Py:",pyinit)
    axt,ayt=F(xinit,yinit)
    xt,yt=[xinit],[yinit]
    vxt,vyt=pxinit/mx,pyinit/my
    Vx=[vxt]
    Vy=[vyt]
    V=[PES(xt[-1],yt[-1])]
    e=V[-1] + 0.5*mx*vxt**2 + 0.5*my*vyt**2
    E=[e]
    i=0
    while xt[-1]<Xend:
        if i>MaxSteps:
            break
        x_new, y_new, vxt, vyt, axt, ayt=step(xt[-1],yt[-1],vxt,vyt,axt,ayt,dt)
        xt.append(x_new)
        yt.append(y_new)
        Vx.append(vxt)
        Vy.append(vyt)
        v=PES(xt[-1],yt[-1])#+0.5*m*(vxt**2+vyt**2)
        V.append(v)
        e=PES(xt[-1],yt[-1]) + 0.5*mx*vxt**2 + 0.5*my*vyt**2
        E.append(e)
        i+=1
    return xt,yt,Vx,Vy,E,V

def run_multT(xinit,yinit,NumTraj=10,Xend=5,steps=1000,dt=0.01,TotE=ET):## running till a condition is met
    Xt=[]
    Yt=[]
    Vxt=[]
    Vyt=[]
    Et=[]
    Vt=[]
    for i in range(NumTraj):
        px,py,v=InitializeP(xinit,yinit,TotE)
        xt,yt,vxt,vyt,E,V=run_Till(xinit,yinit,px,py,Xend,steps,dt)
        Xt.append(xt)
        Yt.append(yt)
        Et.append(E)
        Vt.append(V)
        Vxt.append(vxt)
        Vyt.append(vyt)
    return Xt,Yt,Vt,Et,Vxt,Vyt

def run_multR(XR,YR,NumTraj=10,Xend=5,steps=1000,dt=0.01,TotE=ET):## running multiple traj till a condition is met with random initial positions
    Xt=[]
    Yt=[]
    Vxt=[]
    Vyt=[]
    Et=[]
    Vt=[]
    for i in range(NumTraj):
        xinit,yinit,px,py,v=Initialize(XR,YR,TotE)
        xt,yt,vxt,vyt,E,V=run_Till(xinit,yinit,px,py,Xend,steps,dt)
        Xt.append(xt)
        Yt.append(yt)
        Et.append(E)
        Vt.append(V)
        Vxt.append(vxt)
        Vyt.append(vyt)
    return Xt,Yt,Vt,Et,Vxt,Vyt

from time import time
import os

dt=1e-4#time unit is picosecond
TotalTime=3e-1#time unit is picosecond
Steps=int(TotalTime/dt)
#testing

xinit,yinit,pxinit,pyinit,V_0=Initialize([1.4,1.6],[-0.15,0.15],ET)
T=[dt*i for i in range(Steps+1)]

X,Y,Px,Py,E=run(xinit,yinit,pxinit,pyinit,Steps,dt)

fig=plt.figure()
ax=plt.axes(projection='3d')
ax.plot(X,Y,E)
plt.show()

Xi=np.copy(X)
Yi=np.copy(Y)

T=[dt*i for i in range(Steps+1)]
print(f"Total Energy: {ET/Ec}Kcal/mol")
print(f"Totaltime of integration:{TotalTime}ps\nTime step:{dt}ps")
print(f"Relative error in E={(max(E)-min(E))/min(E)}[maximum deviation in Energy/Initial Total Energy]")
print(f"Initial Conditions:{xinit}\t{yinit}\t{pxinit}\t{pyinit}\t{V_0}")
print(f"Final Conditions:{X[-1]}\t{Y[-1]}\t{Px[-1]}\t{Py[-1]}\t{V_0}")

xl,yl,pxl,pyl=X[-1],Y[-1],Px[-1],Py[-1]
X,Y,Px,Py,E=run(xl,yl,-pxl,-pyl,Steps,dt)
print(f"Initial Conditions:{xl}\t{yl}\t{-pxl}\t{-pyl}")
print(f"Final Conditions:{X[-1]}\t{Y[-1]}\t{Px[-1]}\t{Py[-1]}")
print(f"Error in X,Y,Vx,Vy={xinit-X[-1]}\t{yinit-Y[-1]}\t{pxinit+Px[-1]}\t{pyinit+Py[-1]}")

#print(E,Vx)
#plt.plot(T,E)
plt.plot(X,Y)
plt.plot(Xi,Yi)
plt.show()
#X,Y,Vx,Vy,E=run_multP(0,0,10,Steps,dt)
'''
start_T=time()
Xm,Ym,Vm,Em,Vxm,Vym=run_multR([2.05,2.15],[-1,1],100,8,Steps,dt=dt,TotE=ET)
end_T=time()
print(f"\nTime taken for computing={end_T-start_T}")
ErE=[(max(E)-min(E)) for E in Em]
print(f"maximum error={max(ErE)/418.39} Kcal")
Dir=f"{ET/418.39}_{dt}_TS_test"
isdir=os.path.isdir(Dir)
if not(isdir):
    os.mkdir(Dir)

file=f'{Dir}/full_traj_TS'
for i in range(len(Xm)):
    T=[dt*t for t in range(len(Xm[i]))]
    print(f"writing trajectory {i+1} to: {file+f'_{i+1}.traj'}")
    with open(file+f'_{i+1}.traj','w') as f:
        for j in range(len(Xm[i])):
            f.write(f"{T[j]}\t{Xm[i][j]}\t{Vxm[i][j]}\t{Ym[i][j]}\t{Vym[i][j]}\t{Em[i][j]/418.39}\t{Vm[i][j]/418.39}\n")
print("Finished writing files ",Dir)'''
