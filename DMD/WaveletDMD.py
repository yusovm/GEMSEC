# -*- coding: utf-8 -*-
"""
Created on Sat Jan 25 14:34:55 2020

@author: micha
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from MD_Analysis import Angle_Calc
from Transformations import Transformations

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

half = halftime(Angle_DF)
angle_cossin = cossin(half)
angle_cossin_full = angle_cossin.copy()
angle_cossin_full.drop(angle_cossin_full.tail(1).index,inplace=True)
for i in (range(1024-(angle_cossin_full.shape[0]))):   
    angle_cossin_full = angle_cossin_full.append(angle_cossin.iloc[-2,:])
#T=Transformations(angle_cossin_full)
#Transforms=T.wavelet_transform()

#Define dt based on simulation time step
dt=50*(10^-12)

#Create data matrices X1, X2
f=angle_cossin_full.to_numpy()
f=f[0:-1,:]
X=f.T
X1=X[:,0:-1]
X2=X[:,1:]

#Create x and t domains
xi=np.linspace(min(pd.DataFrame.min(Angle_DF)),max(pd.DataFrame.max(Angle_DF)),f.shape[0])
t=np.linspace(0,f.shape[0],f.shape[0])*dt
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

fig, ax = plt.subplots(3,2,sharex='col', sharey='row')

for i in range(3):
    ax[i,0].plot(X[i,:])
    ax[i,1].plot(X_dmd[i,:])

#plt.plot(np.real(Phi))  
#plt.plot(np.abs(X[0,:]))
#plt.plot(np.abs(X_dmd[0,:]))