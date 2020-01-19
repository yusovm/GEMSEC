# -*- coding: utf-8 -*-
"""
Created on Tue Jan 14 13:23:28 2020

@author: micha
"""

import numpy as np
import pandas as pd
from MD_Analysis import Angle_Calc
from Transformations import Transformations

pdb0="pdbs/WT_295K_200ns_50ps_0_run.pdb"
AC0=Angle_Calc(pdb0)
Angle_DF0=AC0.get_phi_psi()

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

half0 = halftime(Angle_DF0)
angle_cossin = cossin(half0)
T=Transformations(angle_cossin)
Transforms0=T.All_Transforms()
