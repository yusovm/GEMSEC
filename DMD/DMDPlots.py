# -*- coding: utf-8 -*-
"""
Created on Wed Feb  5 12:59:01 2020

@author: micha
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from MD_Analysis import Angle_Calc

#Choose desired pdb
pdb="pdbs/WT_295K_200ns_50ps_0_run.pdb"
#Extract phi, psi angles
AC=Angle_Calc(pdb)
Angle_DF=AC.get_phi_psi()

def cossin(data):
    cols = data.columns
    data = data.to_numpy()
    coss = np.cos(data/180.*np.pi)
    sins = np.sin(data/180.*np.pi)
    
    res=pd.DataFrame()
    for i in range(len(cols)):
        res[cols[i]+"_cos"] = coss[:,i]
        res[cols[i]+"_sin"] = sins[:,i]
    
    return res

def halftime(data):
    dropindex = [1+2*i for i in (range(int(data.shape[0]/2)))]
    return data.drop(dropindex)

#half = halftime(Angle_DF)
angle_cossin = cossin(Angle_DF)
angle_cossin_full = angle_cossin.copy()
angle_cossin_full.drop(angle_cossin_full.tail(1).index,inplace=True)
#for i in (range(1024-(angle_cossin_full.shape[0]))):   
#    angle_cossin_full = angle_cossin_full.append(angle_cossin.iloc[-2,:])

#Define dt based on simulation time step
dt=100*(10**-12)

#Create data matrices X1, X2
f=angle_cossin_full.to_numpy()
ft = np.transpose(f)
for i in np.linspace(0,39,40):
    ft[int(i)]=(ft[int(i)]-np.mean(ft[int(i)]))/np.std(ft[int(i)])
f=np.transpose(ft)
f=f[:-1,:]
X=f.T
X1=X[:,0:-1]
X2=X[:,1:]

#Create x and t domains
xi=np.linspace(np.min(f),np.max(f),f.shape[0])
t=np.linspace(0,f.shape[0],f.shape[0])*dt #+200*10**-9
Xgrid,T=np.meshgrid(xi,t)

#Define r # of truncations, rank truncate data via SVD
r=40
U,S,V=np.linalg.svd(X1,full_matrices=False)
Ur=U[:,:r]
Sr=np.diag(S[:r])
Vr=V.T[:,:r]

#Compute DMD modes and eigenvalues
Atilde=np.conjugate(Ur).T @ X2 @ np.conjugate(Vr) @ np.linalg.inv(Sr) #Koopman operator
D,W=np.linalg.eig(Atilde)
Phi=X2 @ np.conjugate(Vr) @ np.linalg.inv(Sr) @ W #DMD modes
Lambda=D.T
omega=np.log(Lambda)/dt #DMD eigenvalues

#Build DMD solution
x1=X[:,0] #Initial condition
b=np.linalg.lstsq(Phi,x1,rcond=None) #Find b = x1*inv(Phi)
time_dynamics=np.zeros((r,f.shape[0]),dtype="complex")                                                                              
for i in range(f.shape[0]):
    time_dynamics[:,i]=(b[0]*np.exp(omega*t[i]))
X_dmd=np.dot(Phi,time_dynamics) #DMD solution

error=np.sqrt((X_dmd-f.T)**2)

def pltsubtitle(i):
    cs = i % 2 == 0
    pp = int(i/2) % 2 == 0
    num = int(i /4)
    return("cos(" if cs else "sin(") + (r"$\phi_{" if pp else r"$\psi_{") + str(num+1) + ("}$)")

fig, ax = plt.subplots(10,1,sharex='col',sharey='row',num=None,
                       figsize=(15,10),dpi=100,constrained_layout=True)
fig.suptitle('Root Mean Squared Error of DMD Predicted Values')
for i in range(10):
    ax[i].plot(t, error[i,:], color='peru')
    ax[i].set_title(pltsubtitle(i))
    ax[i].set_ylabel('RMSE')
    ax[i].set_xlabel('Time (s)')
fig.savefig("Error 1 Norm.png")

fig, ax = plt.subplots(10,1,sharex='col',sharey='row',num=None,
                       figsize=(15,10),dpi=100,constrained_layout=True)
fig.suptitle('Root Mean Squared Error of DMD Predicted Values')
for i in np.linspace(11,20,10):
    ax[int(i)-11].plot(t, error[int(i)-1,:], color='peru')
    ax[int(i)-11].set_title(pltsubtitle(int(i)-1))
    ax[int(i)-11].set_ylabel('RMSE')
    ax[int(i)-11].set_xlabel('Time (s)')
fig.savefig("Error 2 Norm.png")

fig, ax = plt.subplots(10,1,sharex='col',sharey='row',num=None,
                       figsize=(15,10),dpi=100,constrained_layout=True)
fig.suptitle('Root Mean Squared Error of DMD Predicted Values')
for i in np.linspace(21,30,10):
    ax[int(i)-21].plot(t, error[int(i)-1,:], color='peru')
    ax[int(i)-21].set_title(pltsubtitle(int(i)-1))
    ax[int(i)-21].set_ylabel('RMSE')
    ax[int(i)-21].set_xlabel('Time (s)')
fig.savefig("Error 3 Norm.png")

fig, ax = plt.subplots(10,1,sharex='col',sharey='row',num=None,
                       figsize=(15,10),dpi=100,constrained_layout=True)
fig.suptitle('Root Mean Squared Error of DMD Predicted Values')
for i in np.linspace(31,40,10):
    ax[int(i)-31].plot(t, error[int(i)-1,:], color='peru')
    ax[int(i)-31].set_title(pltsubtitle(int(i)-1))
    ax[int(i)-31].set_ylabel('RMSE')
    ax[int(i)-31].set_xlabel('Time (s)')
fig.savefig("Error 4 Norm.png")

fig2, ax2 = plt.subplots(10,2,sharex='col', sharey='row', num=None, 
                         figsize=(15, 10), dpi=100, constrained_layout=True)
fig2.suptitle('Actual vs. DMD Predicted Values')
for i in range(10):
    ax2[i,0].plot(t, X[i,:], color='brown')
    ax2[i,1].plot(t, X_dmd[i,:], color='cadetblue')
    ax2[i,0].set_ylabel(pltsubtitle(i))
    ax2[i,1].set_ylabel(pltsubtitle(i))
    ax2[i,0].set_xlabel('Time (s)')
    ax2[i,1].set_xlabel('Time (s)')
fig2.savefig("Actual vs. Preidcted 1 Norm.png")
plt.show()

fig2, ax2 = plt.subplots(10,2,sharex='col', sharey='row', num=None, 
                         figsize=(15, 10), dpi=100, constrained_layout=True)
fig2.suptitle('Actual vs. DMD Predicted Values')
for i in np.linspace(11,20,10):
    ax2[int(i)-11,0].plot(t, X[int(i)-1,:], color='brown')
    ax2[int(i)-11,1].plot(t, X_dmd[int(i)-1,:], color='cadetblue')
    ax2[int(i)-11,0].set_ylabel(pltsubtitle(int(i)-1))
    ax2[int(i)-11,1].set_ylabel(pltsubtitle(int(i)-1))
    ax2[int(i)-11,0].set_xlabel('Time (s)')
    ax2[int(i)-11,1].set_xlabel('Time (s)')
fig2.savefig("Actual vs. Preidcted 2 Norm.png")
plt.show()

fig2, ax2 = plt.subplots(10,2,sharex='col', sharey='row', num=None, 
                         figsize=(15, 10), dpi=100, constrained_layout=True)
fig2.suptitle('Actual vs. DMD Predicted Values')
for i in np.linspace(21,30,10):
    ax2[int(i)-21,0].plot(t, X[int(i)-1,:], color='brown')
    ax2[int(i)-21,1].plot(t, X_dmd[int(i)-1,:], color='cadetblue')
    ax2[int(i)-21,0].set_ylabel(pltsubtitle(int(i)-1))
    ax2[int(i)-21,1].set_ylabel(pltsubtitle(int(i)-1))
    ax2[int(i)-21,0].set_xlabel('Time (s)')
    ax2[int(i)-21,1].set_xlabel('Time (s)')
fig2.savefig("Actual vs. Preidcted 3 Norm.png")
plt.show()

fig2, ax2 = plt.subplots(10,2,sharex='col', sharey='row', num=None, 
                         figsize=(15, 10), dpi=100, constrained_layout=True)
fig2.suptitle('Actual vs. DMD Predicted Values')
for i in np.linspace(31,40,10):
    ax2[int(i)-31,0].plot(t, X[int(i)-1,:], color='brown')
    ax2[int(i)-31,1].plot(t, X_dmd[int(i)-1,:], color='cadetblue')
    ax2[int(i)-31,0].set_ylabel(pltsubtitle(int(i)-1))
    ax2[int(i)-31,1].set_ylabel(pltsubtitle(int(i)-1))
    ax2[int(i)-31,0].set_xlabel('Time (s)')
    ax2[int(i)-31,1].set_xlabel('Time (s)')
fig2.savefig("Actual vs. Preidcted 4 Norm.png")
plt.show()

plt.axhline(y=0,color='blue',zorder=-1)
plt.axvline(x=0,color='red',zorder=-2)
plt.scatter(np.real(omega),np.imag(omega), color='saddlebrown',zorder=1)
plt.grid()
plt.title('Real and Imaginary Components of 40 DMD Eigenvalues (\u03C9)',y=1.05)
plt.xlabel('Real')
plt.ylabel('Imaginary')
plt.savefig('Real vs. Imaginary (omega) Norm.png',bbox_inches='tight')
plt.show()

plt.axhline(y=0,color='blue',zorder=-1)
plt.axvline(x=0,color='red',zorder=-2)
plt.scatter(np.real(Lambda),np.imag(Lambda), color='saddlebrown',zorder=1)
plt.grid()
plt.title('Real and Imaginary Components of 40 DMD Eigenvalues (\u03BB)')
plt.xlabel('Real')
plt.ylabel('Imaginary')
plt.savefig('Real vs. Imaginary (Lambda) Norm.png',bbox_inches='tight')