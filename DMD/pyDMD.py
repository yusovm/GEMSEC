# -*- coding: utf-8 -*-
"""
Created on Mon Mar  2 16:43:45 2020

@author: micha
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from MD_Analysis import Angle_Calc
from pydmd import DMD

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

f=angle_cossin_full.to_numpy()

dt=50*(10**-12)

xi=np.linspace(np.min(f),np.max(f),f.shape[0])
t=np.linspace(0,f.shape[0],f.shape[1])*dt #+200*10**-9
Xgrid,T=np.meshgrid(xi,t)

dmd = DMD(svd_rank=40)
dmd.fit(f.T)

xl=np.linspace(0,4000*dt,2000)
yl=range(40)
xlabel,ylabel=np.meshgrid(xl,yl)

#Actual
fig = plt.figure(figsize=(17,6))
plt.pcolor(xl, yl, f.real.T)
plt.yticks([])
plt.title('Actual Data')
plt.colorbar()
plt.show()
fig.savefig("PyDMD Actual Data.png")

#Reconstructed
fig2 = plt.figure(figsize=(17,6))
plt.pcolor(xl, yl, dmd.reconstructed_data.real)
plt.yticks([])
plt.title('Reconstructed Data')
plt.colorbar()
plt.show()
fig2.savefig("PyDMD Reconstructed Data.png")

#Error
fig3 = plt.figure(figsize=(17,6))
plt.pcolor(xl, yl, (np.sqrt(f.T-dmd.reconstructed_data)**2).real)
plt.yticks([])
plt.title('RMSE Error')
plt.colorbar()
plt.show()
fig3.savefig("PyDMD Error.png")

#Eigenvalues
dmd.plot_eigs(show_axes=True, show_unit_circle=True)




