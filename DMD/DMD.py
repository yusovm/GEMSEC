# -*- coding: utf-8 -*-
"""
Created on Sat Nov  9 12:05:07 2019

@author: Michael
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from MD_Analysis import Angle_Calc

pdb = "WT_295K_500ns_50ps_1_run.pdb"

AC = Angle_Calc(pdb)
Angle_DF = AC.get_phi_psi()

dt=50*(10^-12)
f=Angle_DF.to_numpy()
f=f[0:-1,:]
X=f.T
X1=X[:,0:-1]
X2=X[:,1:]
xi=np.linspace(min(pd.DataFrame.min(Angle_DF)),max(pd.DataFrame.max(Angle_DF)),f.shape[0])
t=np.linspace(0,f.shape[0],f.shape[0])*dt
Xgrid,T=np.meshgrid(xi,t)

r=4
U,S,V=np.linalg.svd(X1,full_matrices=False)
Ur=U[:,:r]
Sr=np.diag(S[:r])
Vr=V.T[:,:r]

Atilde=np.conjugate(Ur).T @ X2 @ np.conjugate(Vr) @ np.linalg.inv(Sr)
D,W=np.linalg.eig(Atilde)
Phi=X2 @ np.conjugate(Vr) @ np.linalg.inv(Sr) @ W

Lambda=D.T
omega=np.log(Lambda)/dt

x1=X[:,0]
b=np.linalg.lstsq(Phi,x1,rcond=None)
time_dynamics=np.zeros((r,f.shape[0]),dtype="complex")
for i in range(f.shape[0]):
    time_dynamics[:,i]=(b[0]*np.exp(omega*t[i]))

X_dmd=np.dot(Phi,time_dynamics)

fig, ax = plt.subplots(3,2,sharex='col', sharey='row')

for i in range(3):
    ax[i,0].plot(np.abs(X[i,:]))
    ax[i,1].plot(np.abs(X_dmd[i,:]))

#plt.plot(np.real(Phi))  
#plt.plot(np.abs(X[0,:]))
#plt.plot(np.abs(X_dmd[0,:]))