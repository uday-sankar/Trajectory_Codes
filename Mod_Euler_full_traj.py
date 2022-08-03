#Trajectory evaluation of pyrazoline dissociation in reduced dimension using modified Euler method, with the consideration of 2 angular momentums
#A modified Euler method is used to evaluate the trajectories
#The hamiltonians are used with all the angular moment componentes both of C3 group and N2 rotation over C3group 

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

Ec = 418.39 #energy conversion factor [converts from Kcal/mol to a.m.u*Angstrom/ps^2]

ET = 60*Ec #The total energy of the system in a.m.u*Angstrom/(Picosecond)^2
mu = 16.8 #linear reduced mass 
Ich = 25.147 #moment of inertia of the 3 carbon system 

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
    x,y = uniform(Xrange[0],Xrange[1]),uniform(Yrange[0],Yrange[1])
    V = PES(x,y)
    KE = (TotE-V)#in a.m.u*Angstrom/(Picosecond)^2
    KEx = uniform(0,KE)
    KEy = KE - KEx
    
    sign = uniform(-1,1)
    sign = sign/abs(sign)
    vx = sign*sqrt(KEx*2/mu)
    
    sign = uniform(-1,1)
    sign = sign/abs(sign)
    my = mu*x**2

    I = my*Ich/(my+Ich)
    
    vy = sign*sqrt(KEy*2/I)


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


def step_Euler_mod(xi,yi,vxi,vyi,axi,ayi,dt):#modified Euler, acombination of RK4 and Verlet 
    #step 1
    vx_half = vxi + axi*dt/2
    vy_half = vyi + ayi*dt/2

    x_half = xi + vxi*dt/2 + axi*dt**2/8
    y_half = yi + vyi*dt/2 + ayi*dt**2/8

    Fx_half,Fy_half = F(x_half,y_half)
    my1 = mu*x_half**2
    I1 = my1*Ich/(my1+Ich)
    ax_half = Fx_half/mu + x_half*(I1*vy_half/my1)**2
    ay_half = Fy_half/I1 - 2*I1/my1*vx_half*vy_half/x_half
    
    #step 2
    vxn = vx_half + ax_half*dt/2 
    vyn = vy_half + ay_half*dt/2

    xn = x_half + vx_half*dt/2 + ax_half*dt**2/8
    yn = y_half + vy_half*dt/2 + ay_half*dt**2/8

    Fxn,Fyn = F(xn,yn)
    my2 = mu*xn**2
    I2 = my2*Ich/(my2+Ich)
    axn = Fxn/mu + xn*(vyn*I2/my2)**2
    ayn = Fyn/I2 - 2*I2/my2*vxn*vyn/xn

    #Step 3[Final calculation]
    vxf = vxi + (axi + 2*ax_half + axn)*dt/4
    vyf = vyi + (ayi + 2*ay_half + ayn)*dt/4
    
    xf, yf, axf, ayf = xn, yn, axn, ayn


    return xf, yf, vxf, vyf, axf, ayf

def run_Euler(xinit,yinit,vxinit,vyinit,Steps=1000,dt=0.01):
    Fxinit,Fyinit = F(xinit,yinit)
    #total Moment of inertia I
    my = mu*xinit**2
    I = my*Ich/(my+Ich)
    #
    xt, yt = xinit, yinit
    vxt, vyt = vxinit, vyinit
    axt = Fxinit/mu + xinit*(I*vyt/my)**2
    ayt = Fyinit/I - 2*I/my*vxt*vyt/xinit
    e = PES(xt,yt) + 0.5*mu*vxt**2 + 0.5*I*vyt**2
    Xt, Yt = [xt], [yt]
    Vx = [vxt]
    Vy = [vyt]
    E = [e]
    for i in range(Steps):
        xt, yt, vxt, vyt, axt, ayt = step_Euler_mod(xt,yt,vxt,vyt,axt,ayt,dt)
        myn = mu*xt**2
        In = myn*Ich/(myn+Ich)
        e = PES(xt,yt) + 0.5*mu*vxt**2 + 0.5*In*vyt**2
        Xt.append(xt)
        Yt.append(yt)
        Vx.append(vxt)
        Vy.append(vyt)
        E.append(e)
    return Xt,Yt,Vx,Vy,E

def run_Till(xinit,yinit,vxinit,vyinit,Xend=3,MaxSteps=1000,dt=0.01):
    Fxinit,Fyinit = F(xinit,yinit)
    #total Moment of inertia I
    my = mu*xinit**2
    I = my*Ich/(my+Ich)
    #
    xt, yt = xinit, yinit
    vxt, vyt = vxinit, vyinit
    axt = Fxinit/mu + xinit*(I*vyt/my)**2
    ayt = Fyinit/I - 2*I/my*vxt*vyt/xinit
    e = PES(xt,yt) + 0.5*mu*vxt**2 + 0.5*I*vyt**2
    Xt, Yt = [xt], [yt]
    Vx = [vxt]
    Vy = [vyt]
    E = [e]
    V = [PES(xt,yt)]
    i = 0
    while xt < Xend:
        if i > MaxSteps:
            break
        xt, yt, vxt, vyt, axt, ayt = step_Euler_mod(xt,yt,vxt,vyt,axt,ayt,dt)
        myn = mu*xt**2
        In = myn*Ich/(myn+Ich)
        e = PES(xt,yt) + 0.5*mu*vxt**2 + 0.5*In*vyt**2
        v = PES(xt,yt)
        Xt.append(xt)
        Yt.append(yt)
        Vx.append(vxt)
        Vy.append(vyt)
        E.append(e)
        V.append(v)
        i += 1 
    return Xt,Yt,Vx,Vy,E,V

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

from time import process_time as pt
import os

dt=1e-4#time unit is picosecond
TotalTime=3e-1#time unit is picosecond
Steps=int(TotalTime/dt)

start_t=pt()
Xend = 8
Xm,Ym,Vm,Em,Vxm,Vym = run_multR([1.45,1.55],[16*3.141/180,20*3.141/180],100,8,Steps,dt=dt,TotE=ET)
end_t = pt()
print(f"\nCPU time = {end_t-start_t}")

ErE = [(max(E)-min(E))/ET for E in Em]
print(f"Maximum Relative error in energy={max(ErE)/418.39} Kcal/mol")

Dir = f"TRAJ_ModEul_{ET/418.39}"
isdir = os.path.isdir(Dir)
if not(isdir):
    os.mkdir(Dir)

file = f'{Dir}/pyr'
with open(f'{Dir}/readme.txt','w') as f:
    f.write(f"Pyrazoline denitrogenation trajectroy\nModified Euler method\nC3H6 and N2 contribute to moment of inertia")
    f.write(f"\nENERGY:{ET/418.39} Kcal/mol\nTotal Time of integration:{TotalTime}\nTime step:{dt}\nMaximum Relative error in energy={max(ErE)/418.39} Kcal/mol")
    f.write(f"\nTotal CPU time:{end_t-start_t}\nTotal No of trajectories:100")
    f.write(f"\nThe trajectory starts from the pyrazoline well. The trajectories are stopped whenever the trajectory passes {Xend} or when the timelimit has reached.")
for i in range(len(Xm)):
    T = [dt*t for t in range(len(Xm[i]))]
    print(f"writing trajectory {i+1} to: {file+f'_{i+1}.traj'}")
    with open(file+f'_{i+1}.traj','w') as f:
        for j in range(len(Xm[i])):
            f.write(f"{T[j]}\t{Xm[i][j]}\t{Vxm[i][j]}\t{Ym[i][j]}\t{Vym[i][j]}\t{Em[i][j]/418.39}\t{Vm[i][j]/418.39}\n")
print("Finished writing files ",Dir)
