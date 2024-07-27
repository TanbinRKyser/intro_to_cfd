#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 16 11:57:18 2024

@author: tusker
"""

import numpy as np
import matplotlib.pyplot as plt
import numpy.random as rnd
#%%
N = 20

random_numbers = rnd.rand(N) # [0-1] nummbers
#print(random_numbers)

N = 30000
random_numbers = rnd.uniform( -3.0, 5.0, N ) # -3 < r < 5 
#print(random_numbers)

plt.figure( figsize=(10,6) )
#plt.hist( random_numbers, bins=8 )
plt.hist( random_numbers, bins=8, density=True )
#plt.axhline( 0.13, c='b', ls='--' ).axhline( 0.13, c='b', ls='--' )
plt.axhline( 1.0/8.0, c='b', ls='--' )
#%%
# Task 3
# LCG
dice_rolls = rnd.randint( 1, 7, 5000 )

plt.figure( figsize=(10,6) )
#plt.hist( dice_rolls)
plt.hist( dice_rolls, bins=[0.5,1.5,2.5,3.5,4.5,5.5,6.5], density=True )
plt.axhline( 1.0/6.0, c='k', ls='--' )


def mean(x):
    mean = 0.
    for i in range( len(x) ):
         mean += x[i]
    mean /= len(x)
    return mean

def stddevUnbiased(x):
    var = 0.
    meanX = mean(x)
    N = len(x)
    for i in range(N):
        var += (x[i] - meanX)**2
    var /= N-1
    return np.sqrt(var)

print('mean = %1.6f stddev = %1.6f'%(mean(dice_rolls),stddevUnbiased( dice_rolls )))

#%%
# Task 4
# CDF inversion

t = np.linspace( 0.0, 100, 1000)
refPDF = np.exp( -t )

r = rnd.uniform( 0.0, 1.0, 50000 )

ts = - np.log( 1.0 - r )

plt.figure(figsize=(10,6))
#plt.plot(ts, np.ones(len(ts)), 'k.')
#plt.plot(ts,r,'k.')
#plt.plot(t, 1-np.exp(-t),'r--')

#plt.hist( ts, bins = np.linspace( 0.0, 8.0, 200 ), density=True )
plt.hist( ts, bins = np.linspace( 0.0, 8.0, 20 ), density=True )
plt.plot( t, refPDF,'k--' )

plt.xlim([0.0,8.0])

plt.show()