# -*- coding: utf-8 -*-
"""
Created on Fri Feb 21 12:56:53 2020

@author: micha
"""

import numpy as np
import pandas as pd
import os
import sys
import scipy as scp
import matplotlib.pyplot as plt

class Adj_Mats():
    def __init__(self, pdb):
        self.file = pdb
        self.d_graphs = np.zeros(1, int)
        self.a_graphs = np.zeros(1, int)
    def set_AtomDists(self, new_dists):
        self.d_graphs = new_dists
    def set_AtomAdj(self, new_adj):
        self.a_graphs = new_adj
    #Adapted from: xplo2xyz.py on Open NMR Project (http://nmrwiki.org/wiki/index.php?title=Script_xplo2xyz.py_-_convert_PDB_files_created_by_NIH-XPLOR_to_XYZ_file_format)
    def get_AtomDists(self):
        class PDBAtom(object):
            def __init__(self, string):
                #this is what we need to parse
                #ATOM      1  CA  ORN     1       4.935   1.171   7.983  1.00  0.00      sega
                #XPLOR pdb files do not fully agree with the PDB conventions
                self.name = string[12:16].strip()
                self.x = float(string[30:38].strip())
                self.y = float(string[38:46].strip())
                self.z = float(string[46:54].strip())
                self.warnings = []
                if len(string) < 78:
                    self.element = self.name[0]
                    self.warnings.append('Chemical element name guessed ' +\
                                         'to be %s from atom name %s' % (self.element, self.name))
                else:
                    self.element = string[76:78].strip()
        if os.path.isfile(self.file):
            pdb_file = open(self.file,'r')
        else:
            raise Exception('file %s does not exist' % self.file)
        lineno = 0
        frames = []
        atoms = []
        #read pdb file
        for line in pdb_file:
            lineno += 1
            if line.startswith('ATOM'):
                try:
                    at_obj = PDBAtom(line)
                    atoms.append([at_obj.x, at_obj.y, at_obj.z])
                except:
                    sys.stderr.write('\nProblem parsing line %d in file %s\n' % (lineno,self.file))
                    sys.stderr.write(line)
                    sys.stderr.write('Probably ATOM entry is formatted incorrectly?\n')
                    sys.stderr.write('Please refer to - http://www.wwpdb.org/documentation/format32/sect9.html#ATOM\n\n')
                    sys.exit(1)
            elif line.startswith('END'):
                frames.append(atoms)
                atoms = []
        pdb_file.close()
        base = np.zeros((len(frames), len(frames[0]), 3))
        for i in range(len(frames)):
            for j in range(len(frames[i])):
                for k in range(len(frames[i][j])):
                    base[i][j][k] = frames[i][j][k]
        dists = np.reshape(base, (len(frames), 1, len(frames[0]), 3)) - np.reshape(base, (len(frames), len(frames[0]), 1, 3))
        dists = dists**2
        dists = dists.sum(3)
        dists = np.sqrt(dists)
        self.d_graphs = dists
        return self.d_graphs
    #Parameter:
    #   -t: The threshold distance for adjacency in Angstroms (4-25)
    def get_AtomAdj(self, t = 4):
        if len(self.d_graphs) == 1:
            self.get_AtomDists()
        self.a_graphs = (self.d_graphs < t).astype(int)
        return self.a_graphs

pdb="pdbs/WT_295K_200ns_50ps_0_run.pdb"
Dist=Adj_Mats(pdb).get_AtomDists()

DistT = np.zeros(((len(Dist[0]) * (len(Dist[0]) + 1))//2, len(Dist)))
for i, sqr_mat in enumerate(Dist):
    a = 0
    for j in range(len(sqr_mat)):
        for k in range(len(sqr_mat)):
            if j < k:
                continue
            else:
                DistT[a][i] = sqr_mat[j][k]
                a += 1

dt=50*(10**-12)

#Create data matrices X1, X2
f=DistT
#ft = np.transpose(f)
#for i in np.linspace(0,2000,2001):
#    ft[int(i)]=(ft[int(i)]-np.mean(ft[int(i)]))/np.std(ft[int(i)])
#f=np.transpose(ft)
f=f[:-1,:]
X=f
X1=X[:,:-1]
X2=X[:,1:]

#Create x and t domains
xi=np.linspace(np.min(f),np.max(f),f.shape[0])
t=np.linspace(0,f.shape[0],f.shape[0])*dt #+200*10**-9
Xgrid,T=np.meshgrid(xi,t)

#Define r # of truncations, rank truncate data via SVD
r=100
U,S,V=scp.linalg.svd(X1,full_matrices=False)
Ur=U[:,:r]
Sr=np.diag(S[:r])
Vr=V.T[:,:r]

#Compute DMD modes and eigenvalues
Atilde=Ur.T @ X2 @ Vr @ scp.linalg.inv(Sr) #Koopman operator
D,W=np.linalg.eig(Atilde)
Phi=X2 @ np.conjugate(Vr) @ scp.linalg.inv(Sr) @ W #DMD modes
Lambda=D
omega=np.log(Lambda)/dt #DMD eigenvalues

#Build DMD solution
x1=X[:,0] #Initial condition
b=np.linalg.lstsq(Phi,x1,rcond=None) #Find b = x1*inv(Phi)
time_dynamics=np.zeros((r,f.shape[0]),dtype="complex")                                                                              
for i in range(f.shape[0]):
    time_dynamics[:,i]=(b[0]*np.exp(omega*t[i]))
X_dmd=np.dot(Phi,time_dynamics) #DMD solution
