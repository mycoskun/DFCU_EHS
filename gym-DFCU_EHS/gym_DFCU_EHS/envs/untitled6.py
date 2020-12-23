# -*- coding: utf-8 -*-
"""
Created on Sun Dec 20 21:54:12 2020

@author: MYCoskun
"""

import numpy as np

valve = np.arange(0,5)

c = np.array(np.meshgrid(valve,valve)).T.reshape(-1,2)
d = np.zeros((len(c),2))

action_1 = np.concatenate((c[:,0].reshape(-1,1),d[:,0].reshape(-1,1),
                               d[:,0].reshape(-1,1),c[:,1].reshape(-1,1)), 
                              axis=1)

action_2 = np.concatenate((d[:,0].reshape(-1,1),c[:,0].reshape(-1,1),
                               c[:,1].reshape(-1,1),d[:,0].reshape(-1,1)),
                              axis=1)

action_space = np.vstack((action_1,action_2))