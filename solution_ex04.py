#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 11 10:30:42 2024

@author: tusker
"""

import numpy as np
import matplotlib.pyplot as plt

#%%
#Task 1(a):
def gauss( x, mu = 0.3, sigma = 1.2 ):
    return np.exp( - ( x - mu )**2 / ( 2 * sigma**2 ) ) / np.sqrt( 2 * np.pi * sigma**2 )

def gaussDerivative( x, mu=0.3, sigma = 1.2 ):
    return -( x - mu ) / sigma**2 * np.exp( -( x - mu )**2 / ( 2 * sigma**2 ) ) / np.sqrt( 2 * np.pi * sigma**2 )

#%%

def forwardDeriv( y, h ):
    z = np.zeros_like( y )
    for i in range( 0, len( y ) - 1, 1 ):
        z[i]= ( y[i+1] - y[i] ) / h
    
    return z
#%%
def centralDeriv( y, h ):
    
    z = np.zeros_like( y )
    
    for i in range(1,len(y)-1,1):
        z[i] = ( y[i+1] - y[i-1] ) / (2*h)
    
    return z
#%%

x_min = -5.0
x_max = 5.0

h = 10**-2
N = int( ( x_max - x_min ) / h )
print( N )

N = 11
#x = np.linspace( x_min, x_max, N+1 )
x,h = np.linspace( -5.0, 5.0, N, retstep=True )
y = gauss( x )

plt.figure(figsize=(12,6))

plt.plot( x, y, 'b-' )

plt.plot( x, gaussDerivative( x ), 'r-' )
plt.plot( x,forwardDeriv( y, h ), 'r.' )
plt.plot( x,centralDeriv( y, h ), 'r+' )