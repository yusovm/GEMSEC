# -*- coding: utf-8 -*-
"""
Created on Sat Dec 12 12:43:41 2020

@author: micha
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

error1=np.load("error1.data",allow_pickle=True)
error1_plot=np.zeros((36500,100))
for i in range(len(error1)):
    error1_plot[:len(error1[i]),i]=error1[i]
fig = plt.figure(figsize=(10,10))
ax = fig.add_subplot(111)
img = ax.imshow(error1_plot,origin='lower',extent=[50, 36500, 50, 36500],
           cmap=cm.RdYlGn,aspect='auto',)
ax.set_xlabel('# of time steps used')
ax.set_ylabel('# of time steps reconstructed/predicted')
fig.colorbar(img)
fig.savefig(fname='errorplot1.png')

error2=np.load("error2.npy")
t_num_samples = [int(x) for x in np.linspace(50,36500,100)]
t_num_samples_num = len(t_num_samples)
t_timestep = [int(x) for x in np.linspace(1,730,100)]
e_idxes =[i for i in np.ndindex(t_num_samples_num, t_num_samples_num, len(t_timestep))]
X = np.zeros(len(e_idxes))
Y = np.zeros(len(e_idxes))
Z = np.zeros(len(e_idxes))
E = np.zeros(len(e_idxes))

j = 0
for i in range(len(e_idxes)):
    xi,yi,zi = e_idxes[i]
    e = error2[xi,yi,zi]
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
img = ax.scatter(X, Y, Z, c=E, s=0.01)
fig.colorbar(img)
ax.set_xlabel('# of time steps used')
ax.set_ylabel('# of time steps reconstructed/predicted')
ax.set_zlabel('sampling sparsity (# total time steps/# time steps used)')
plt.show()
for angle in range(0, 360):
    ax.view_init(30, angle)
    fig.savefig('ang'+str(angle)+'.png')