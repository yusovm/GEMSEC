# -*- coding: utf-8 -*-
"""
Created on Sat Oct 31 12:17:28 2020

@author: micha
"""

import numpy as np
import pandas as pd
import pywt
import sys
import pickle
from pywt import wavedec
from MD_Analysis import Angle_Calc

#Import pdb files and group them by 50ps/100ps
pdb1="pdbs/WT_295K_200ns_50ps_0_run.pdb"
pdb2="pdbs/WT_295K_500ns_50ps_1_run.pdb"
pdb3="pdbs/WT_295K_500ns_50ps_2_run.pdb"
pdb4="pdbs/WT_295K_500ns_50ps_3_run.pdb"
pdb5="pdbs/WT_295K_500ns_50ps_4_run.pdb"
pdb6="pdbs/WT_295K_500ns_50ps_5_run.pdb"
pdb7="pdbs/WT_295K_500ns_100ps_6_run.pdb"
pdb8="pdbs/WT_295K_500ns_100ps_7_run.pdb"
pdb9="pdbs/WT_295K_500ns_100ps_8_run.pdb"
pdb10="pdbs/WT_295K_500ns_100ps_9_run.pdb"
pdb11="pdbs/WT_295K_300ns_100ps_10_run.pdb"
pdb50 = [pdb1,pdb2,pdb3,pdb4,pdb5,pdb6]
pdb100 = [pdb7,pdb8,pdb9,pdb10,pdb11]

#Extract phi/psi angles
pdb50_ang = [Angle_Calc(i).get_phi_psi().drop([0]) for i in pdb50]
pdb100_ang = [Angle_Calc(i).get_phi_psi().drop([0]) for i in pdb100]

def drop_every_second_row(x):
    return x.iloc[[i % 2 == 0 for i in range(x.shape[0])],:]

#Drop every other row from 50ps runs to obtain 100ps data and combine pdb data
pdb50_ang_2 = [drop_every_second_row(i) for i in pdb50_ang]
Angle_DF = pd.concat(pdb50_ang_2 + pdb100_ang)

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

#Compute cosine/sine and normalize
angle_cossin = cossin(Angle_DF)
f=angle_cossin.to_numpy()
ft = np.transpose(f)
# for i in np.linspace(0,39,40):
#     ft[int(i)]=(ft[int(i)]-np.mean(ft[int(i)]))/np.std(ft[int(i)])
# f=np.transpose(ft)

#Define dt based on simulation time step (100ps) and r based on number of columns
dt=100*(10**-12)
r=40

#DMD function (rows<columns)
def DMD(dt,f,r):
    #Create data matrices X1, X2
    X=f.T
    X1=X[:,0:-1]
    X2=X[:,1:]

    #Create x and t domains
    xi=np.linspace(np.min(f),np.max(f),f.shape[0])
    t=np.linspace(0,f.shape[0],f.shape[0])*dt #+200*10**-9
    Xgrid,T=np.meshgrid(xi,t)

    #Define r # of truncations, rank truncate data via SVD
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
    return X_dmd

def DMDError_2Wn(data,x,dt):
    f = data[0:x,:]
    r = f.shape[1]
    index=int(np.log2(f.shape[0]))
    coeffs={}
    wavDMD={}
    error={}
    for wavname in pywt.wavelist(kind='discrete'):
        print('Working on ' + wavname)
        coeffs[wavname] = wavedec(f, wavname, level=index-1, axis=0)
        try:
            wavDMD[wavname]=[0]*index
            for i in range(0,index):
                if coeffs[wavname][i].shape[0]>r:
                    wavDMD[wavname][i]=DMD(dt,coeffs[wavname][i],r).T
                else:
                    wavDMD[wavname][i]=DMD(dt,coeffs[wavname][i],coeffs[wavname][i].shape[0]-1).T
            invf=pywt.waverec(wavDMD[wavname], wavname,axis=0)
            error[wavname]=np.sqrt(np.mean((invf[:x,:]-f)**2))
        except:
            print("Unexpected error while working on:" + wavname, sys.exc_info()[0])
    bestwavelet=''
    for wavname in error:
        errorlast=error[wavname]
        errorbest=error.get(bestwavelet,np.Inf)
        if errorlast<errorbest:
            bestwavelet=wavname
        print("Best wavelet: " + bestwavelet + " with error "+str(errorbest))
    bestinvf=pywt.waverec(wavDMD[bestwavelet],bestwavelet,axis=0)
    besterror=np.zeros(x)
    for n in range(0,x):
        besterror[n]=np.sqrt(np.mean((bestinvf[:(n+1),]-f[:(n+1),:])**2))
    return besterror

def DMDError_3W(data,num_timesteps,timestep,dt):
    print(num_timesteps, timestep)
    f = data[0:num_timesteps,:]
    r = f.shape[1]
    index=int(np.log2(f.shape[0]))
    sample_indexes = [i % timestep == 0 for i in range(f.shape[0])]
    f = f[sample_indexes,:]
    coeffs={}
    wavDMD={}
    error={}
    for wavname in pywt.wavelist(kind='discrete'):
        print('Working on ' + wavname)
        coeffs[wavname] = wavedec(f, wavname, level=index-1, axis=0)
        try:
            wavDMD[wavname]=[0]*index
            for i in range(0,index):
                if coeffs[wavname][i].shape[0]>r:
                    wavDMD[wavname][i]=DMD(1,coeffs[wavname][i],r).T
                else:
                    wavDMD[wavname][i]=DMD(1,coeffs[wavname][i],coeffs[wavname][i].shape[0]-1).T
            invf=pywt.waverec(wavDMD[wavname], wavname,axis=0)
            error[wavname]=np.sqrt(np.mean((invf[:num_timesteps,:]-f)**2))
        except:
            print("Unexpected error while working on:" + wavname, sys.exc_info()[0])
    bestwavelet=''
    for wavname in error:
        errorlast=error[wavname]
        errorbest=error.get(bestwavelet,np.Inf)
        if errorlast<errorbest:
            bestwavelet=wavname
        print("Best wavelet: " + bestwavelet + " with error "+str(errorbest))
    bestinvf=pywt.waverec(wavDMD[bestwavelet],bestwavelet,axis=0)
    besterror=np.zeros(int(num_timesteps/timestep))
    for n in range(0,int(num_timesteps/timestep)):
        besterror[n]=np.sqrt(np.mean((bestinvf[:(n+1),:]-f[:(n+1),:])**2))
        return besterror

print("Starting calculation #1")
tstep = [int(x) for x in np.linspace(2,36500,100)]
error_data1 = [DMDError_2Wn(f, i,dt) for i in tstep]
with open('error1.data', 'wb') as filehandle:
    pickle.dump(error_data1, filehandle)

print("Starting calculation #2")
t_num_samples = [int(x) for x in np.linspace(2,36500,100)]
t_timestep = [int(x) for x in np.linspace(1,3650,100)]
error_data = np.zeros((len(t_num_samples),len(t_timestep),36500))
for i in range(len(t_num_samples)):
    for j in range(len(t_timestep)):
        try:
            if t_num_samples[i]/t_timestep[j]>10:
                print("Working on: " + str(i) + " samples with " + str(j) + " sparsity")
                err = DMDError_3W(f, t_num_samples[i], t_timestep[j],dt)
                if type(err)==type(None):
                    print("Not enough data")
                    continue
                error_data[i,j,:len(err)] = err
        except:
            print("Unexpected error while working on:" + str(i) + " samples with " + str(j) + " sparsity", sys.exc_info()[0])
np.save("error2",error_data)
np.save("error2numsamples",t_num_samples)
np.save("error2timestep",t_timestep)
