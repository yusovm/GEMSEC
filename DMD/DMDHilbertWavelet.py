# -*- coding: utf-8 -*-
"""
Created on Sat Oct 31 12:17:28 2020

@author: micha
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import pywt
import csv
import sys
from pywt import wavedec
from scipy.signal import hilbert
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
for i in np.linspace(0,39,40):
    ft[int(i)]=(ft[int(i)]-np.mean(ft[int(i)]))/np.std(ft[int(i)])
f=np.transpose(ft)

#Extend matrix size with last data point repeated
for i in range(f.shape[0]):
    index=i
    if 2**index>f.shape[0]:
        break 
fadd=np.zeros((2**index-int(f.shape[0]), int(f.shape[1])))
for i in range(fadd.shape[0]):
    fadd[i]=f[-1]
fwav=np.vstack((f,fadd))

fs = 1
signal = f[:,0]
analytic_signal = hilbert(signal)
amplitude_envelope = np.abs(analytic_signal)
instantaneous_phase = np.unwrap(np.angle(analytic_signal))
instantaneous_frequency = (np.diff(instantaneous_phase) / (2.0*np.pi) * fs)
fig = plt.figure()
ax0 = fig.add_subplot(211)
ax0.plot(signal, label='signal')
ax0.plot(amplitude_envelope, label='envelope')
ax0.set_xlabel("time in seconds")
ax0.legend()
ax1 = fig.add_subplot(212)
ax1.plot(instantaneous_frequency)
ax1.set_xlabel("time in seconds")

analytic_signalf = hilbert(f,axis=0)
amplitude_envelopef = np.abs(analytic_signalf)
instantaneous_phasef = np.unwrap(np.angle(analytic_signalf))
instantaneous_frequencyf = (np.diff(instantaneous_phasef) / (2.0*np.pi) * fs)

#Compute wavelet transform
(cA, cD)=pywt.dwt(amplitude_envelopef, 'haar', axis=0)
cA.shape
cD.shape
coeffs = wavedec(amplitude_envelopef, 'haar', level=15, axis=0)

#Separate coefficients by different frequencies, run DMD
freq=[0]*index
freq[0]=cA
for i in np.linspace(1,index-1,index-1):
    freq[int(i)]=cD[int(cD.shape[0]*(2**(i-1)-1)/(2**(i-1))):int(cD.shape[0]*(2**(i)-1)/(2**(i))),:]
wavDMD=[0]*index
for i in range(0,index):
    if freq[i].shape[0]>40:
        wavDMD[i]=DMD(1,freq[i],40)
    else:
        wavDMD[i]=DMD(1,freq[i],freq[i].shape[0]-1)

#Reconstruct cA, cD, and compute inverse transform and average RMSE
cAinv=wavDMD[0].T
cDinv=np.concatenate([x.T for x in wavDMD[1:]], axis = 0)
cDinv=np.vstack((cDinv,cDinv[-1]))
invf=pywt.idwt(cAinv,cDinv,'db1',axis=0)
error=np.mean(np.sqrt((invf-fwav)**2))

wavDMD=[0]*index
coeffs = wavedec(amplitude_envelopef, 'haar', level=index-1, axis=0)
for i in range(0,index):
    if coeffs[i].shape[0]>40:
        wavDMD[i]=DMD(1,coeffs[i],40).T
    else:
        wavDMD[i]=DMD(1,coeffs[i],coeffs[i].shape[0]-1).T
invf=pywt.waverec(wavDMD, 'haar',axis=0)
error=np.mean(np.sqrt((invf-fwav)**2))
error

coeffs={}
wavDMD={}
error1={}
for wavname in pywt.wavelist(kind='discrete'):
    print('Working on ' + wavname)
    coeffs[wavname] = wavedec(analytic_signalf, wavname, level=index-1, axis=0)
    try:
        wavDMD[wavname]=[0]*index
        for i in range(0,index):
            if coeffs[wavname][i].shape[0]>40:
                wavDMD[wavname][i]=DMD(1,coeffs[wavname][i],40).T
            else:
                wavDMD[wavname][i]=DMD(1,coeffs[wavname][i],coeffs[wavname][i].shape[0]-1).T
        invf=pywt.waverec(wavDMD[wavname], wavname,axis=0)
        error1[wavname]=np.mean(np.sqrt((invf-analytic_signalf)**2))
    except:
        print('Exception while working on ' + wavname)
        
dict = {'Python' : '.py', 'C++' : '.cpp', 'Java' : '.java'}
w = csv.writer(open("errorhws.csv", "w"))
for key, val in error1.items():
    w.writerow([key, val])
    
X_dmd_hilbert = DMD(1,analytic_signalf,40)
invfh = -hilbert(X_dmd_hilbert.real,axis=1)
error2=np.sqrt(np.mean(np.abs((invfh-f.T))**2))
        
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

#Applying DMD with varying amounts of data used and calculating average error
def DMDError_1(data,x):
    f = data[0:x,:]
    X_dmd = DMD(dt,f,r)
    error=np.sqrt((X_dmd-f.T)**2)
    return np.mean(error)

#Applying DMD with varying amounts of data used and reconstructed/predicted and calculating average error
def DMDError_2(data,x,n):
    f = data[0:x,:]
    X_dmd = DMD(dt,f,r)
    X_dmd = X_dmd[:,0:n]
    error=np.mean((X_dmd-f.T[:,0:n])**2)
    return np.sqrt(error)

#Applying DMD with varying amounts of data used, reconstructed/predicted, and sparser sampling
#while changing truncation when less columns than rows and calculating average error
def DMDError_3(data,num_timesteps,num_samples_recon,timestep):
    print(num_timesteps, num_samples_recon, timestep)
    g = data[0:num_timesteps,:]
    sample_indexes = [i % timestep == 0 for i in range(g.shape[0])]
    g = g[sample_indexes,:]
    if num_timesteps/timestep > f.shape[1]:
        r = f.shape[1]
    else: 
        r = int(np.floor(num_timesteps/timestep))
    X_dmd = DMD(dt,g,r)
    X_dmd = X_dmd[:,0:num_samples_recon]
    error=np.sqrt((X_dmd-g.T[:,0:num_samples_recon])**2)
    return np.mean(error)
    
#Applying DMD with varying amounts of data used and reconstructed/predicted and calculating average error
def DMDError_2H(data,x,n,dt,r):
    f = data[0:x,:]
    f_hilbert = hilbert(f,axis=0)
    X_dmd = DMD(dt,f_hilbert,r)
    X_dmd = -hilbert(X_dmd,axis=0)
    X_dmd = X_dmd[:,0:n]
    error=np.sqrt((X_dmd-f.T[:,0:n])**2)
    return np.mean(error)

def DMDError_2W(data,x,n,dt,r):
    if n>x:
        return np.nan
    f = data[0:x,:]
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
                if coeffs[wavname][i].shape[0]>40:
                    wavDMD[wavname][i]=DMD(1,coeffs[wavname][i],40).T
                else:
                    wavDMD[wavname][i]=DMD(1,coeffs[wavname][i],coeffs[wavname][i].shape[0]-1).T
            invf=pywt.waverec(wavDMD[wavname], wavname,axis=0)
            invf=invf[:n,:]
            error[wavname]=np.sqrt(np.mean((invf-f[:n,:])**2))
        except:
            print("Unexpected error while working on:" + wavname, sys.exc_info()[0])
    return min(error.values())

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

def DMDError_3(data,num_timesteps,num_samples_recon,timestep):
    print(num_timesteps, num_samples_recon, timestep)
    g = data[0:num_timesteps,:]
    sample_indexes = [i % timestep == 0 for i in range(g.shape[0])]
    g = g[sample_indexes,:]
    if num_timesteps/timestep > f.shape[1]:
        r = f.shape[1]
    else: 
        r = int(np.floor(num_timesteps/timestep))
    X_dmd = DMD(dt,g,r)
    X_dmd = X_dmd[:,0:num_samples_recon]
    error=np.sqrt((X_dmd-g.T[:,0:num_samples_recon])**2)
    return np.mean(error)

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

tstep = [int(x) for x in np.linspace(2,36500,100)]
error_data = [DMDError_2Wn(f, i,dt) for i in tstep]
np.save
error_plot_data = np.real(np.array(error_data))


plt.figure(figsize = (10,10))
plt.imshow(error_plot_data,origin='lower',extent=[50, 36500, 50, 36500],
           cmap=cm.RdYlGn,aspect='auto')
plt.xlabel('# of time steps used')
plt.ylabel('# of time steps reconstructed/predicted')
plt.colorbar()

t_num_samples = [int(x) for x in np.linspace(50,36500,100)]
t_num_samples_num = len(t_num_samples)
t_timestep = [int(x) for x in np.linspace(1,730,100)]
error_data = np.zeros((t_num_samples_num,t_num_samples_num,len(t_timestep)))
for i in range(t_num_samples_num):
    for j in range(t_num_samples_num):
        for k in range(len(t_timestep)):
            error_data[i,j,k] = DMDError_3(f, 
                          t_num_samples[i], t_num_samples[j], t_timestep[k])
np.save("errors_fourier",error_data)

e_idxes =[i for i in np.ndindex(t_num_samples_num, t_num_samples_num, len(t_timestep))]
X = np.zeros(len(e_idxes))
Y = np.zeros(len(e_idxes))
Z = np.zeros(len(e_idxes))
E = np.zeros(len(e_idxes))

j = 0
for i in range(len(e_idxes)):
    xi,yi,zi = e_idxes[i]
    e = error_data[xi,yi,zi]
    if e < 10E-6 or e > 1:
        continue
    X[j] = t_num_samples[xi]
    Y[j] = t_num_samples[yi]
    Z[j] = t_timestep[zi]
    E[j] = e
    j = j + 1
    
X = X[0:j]
Y = Y[0:j]
Z = Z[0:j]
E = E[0:j]
    
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
img = ax.scatter(X, Y, Z, c=E, s=1)
fig.colorbar(img)
ax.set_xlabel('# of time steps used')
ax.set_ylabel('# of time steps reconstructed/predicted')
ax.set_zlabel('sampling sparsity (# total time steps/# time steps used)')
plt.show()
for angle in range(0, 360):
    ax.view_init(30, angle)
    fig.savefig('ang'+str(angle)+'.png')
    
def pltsubtitle(i):
    cs = i % 2 == 0
    pp = int(i/2) % 2 == 0
    num = int(i /4)
    return("cos(" if cs else "sin(") + (r"$\phi_{" if pp else r"$\psi_{") + str(num+1) + ("}$)")
 
t=np.linspace(0,36500,36500)   
 
fig2, ax2 = plt.subplots(10,2,sharex='col', sharey='row', num=None, 
                        figsize=(15, 10), dpi=100, constrained_layout=True)
fig2.suptitle('Actual vs. Wavelet-DMD Predicted Values')
for i in range(10):
    ax2[i,0].plot(t, f[:,i], color='brown')
    ax2[i,1].plot(t, invf[:,i], color='cadetblue')
    ax2[i,0].set_ylabel(pltsubtitle(i))
    ax2[i,1].set_ylabel(pltsubtitle(i))
    ax2[i,0].set_xlabel('Time (s)')
    ax2[i,1].set_xlabel('Time (s)')
fig2.savefig("waveletexample.png")
plt.show()

waverror=np.zeros(len(wavDMD['coif16']))
for i in range(len(wavDMD['coif16'])):
    waverror[i]=np.sqrt(np.mean((coeffs['coif16'][i]-wavDMD['coif16'][i])**2))