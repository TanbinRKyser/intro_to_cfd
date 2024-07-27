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
#%%
#Task 1(b):
def gaussDerivative( x, mu=0.3, sigma = 1.2 ):
    #return -( x - mu ) / sigma**2 * np.exp( -( x - mu )**2 / ( 2 * sigma**2 ) ) / np.sqrt( 2 * np.pi * sigma**2 )
    return -( x - mu ) / sigma**2 * gauss( x, mu, sigma )
#%%
# Task 1(d):
def gaussSecondDerivative( x, mu=0.3, sigma=1.2):
    return ( ( (x-mu)**2 - sigma**2 ) / sigma**4 ) * gauss( x, mu, sigma )
#%%

def forwardDeriv( y, h ):
    z = np.zeros_like( y )
    for i in range( 0, len( y ) - 1, 1 ):
        z[i]= ( y[i+1] - y[i] ) / h
    
    return z
#%%
# Task 1(c):
def centralDeriv( y, h ):
    
    z = np.zeros_like( y )
    
    #for i in range( 1, len(y) - 1, 1 ):
        #z[i] = ( y[i+1] - y[i-1] )
    z[1:-1] = ( y[2:] - y[0:-2] ) / ( 2*h )
    
    return z
#%%
# discretize directly
def centralSecondDeriv( y, h ):
    
    z = np.zeros_like( y )
    
    #for i in range( 1, len(y) - 2, 1 ):
        #z[i] = ( y[i+1] - 2 * y[i] + y[i-1] )
    z[1:-1] = ( y[2:] - 2 * y[1:-1] + y[0:-2] ) / (  h**2 )
    
    return z

#%%

x_min = -5.0
x_max = 5.0

h = 10**-2
#N = int( ( x_max - x_min ) / h )
#print( 'N: ',N )

#N = 10
N = 100
#x = np.linspace( x_min, x_max, N+1 )
x,h = np.linspace( x_min, x_max, N+1, retstep=True )
#print('h: %1.3f'%h)
y = gauss( x )

plt.figure(1, figsize=(10,6) )

plt.plot( x, y, 'b-', label='$g(x)$' )

y_gd = gaussDerivative( x )
plt.plot( x, y_gd, 'r--', label ="$g'(x)$" )

#plt.plot( x, forwardDeriv( y, h ), 'r.', label='Forward_gauss_first' )
y_cd = centralDeriv( y, h )
plt.plot( x, y_cd, 'g--', label="$g'_z(x)" )

y_gsd = gaussSecondDerivative(x)
plt.plot( x, y_gsd, 'g:', label ="$g''_z(x)" )

y_csd = centralSecondDeriv(y, h)
plt.plot( x, y_csd, 'r:', label="$g''_z(x)$" )

# alternative
tmp = y_cd[1:-1]
y_cdcd = centralDeriv( tmp, h )
plt.plot( x[1:-1], y_cdcd,'r-.', label="$g''_zz(x)$" )

plt.grid()
plt.legend( loc='best' )
plt.show()
#%%

"""def epsilon_1( x, h ):
    y = gauss(x)
    return centralDeriv( y, h) - gaussDerivative(x)

    
def epsilon_2( x, h ):
    y = gauss(x)
    return centralSecondDeriv( y, h) - gaussSecondDerivative(x)

plt.figure(figsize=(10,6))
plt.subplot(1,2,1)
plt.plot(x,epsilon_1(x, h), label='epsilon_1')
plt.plot(x,epsilon_2(x, h), label='epsilon_2')
plt.title('linear')
plt.legend()
plt.grid()

plt.subplot(1,2,2)
plt.semilogy(x,epsilon_1(x, h), label='epsilon_1')
plt.semilogy(x,epsilon_2(x, h), label='epsilon_2')
plt.title('semi-log')
plt.legend()
plt.grid()

plt.show()"""

error1 = np.abs( y_cd - y_gd )
error2 = np.abs( y_csd - y_gsd )

plt.figure( 2, figsize=( 18, 8 ) )
plt.subplot( 1, 2, 1 )
plt.plot( x, error1, label='error1' )
plt.plot( x, error2, label='error2' )
plt.grid()
plt.legend()

plt.subplot( 1, 2, 2 )
plt.semilogy( x, error1, label='error1' )
plt.semilogy( x, error2, label='error2' )
plt.grid()
plt.legend()

plt.show()
