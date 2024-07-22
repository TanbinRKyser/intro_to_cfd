#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 18 14:45:53 2024

@author: tusker
"""

import numpy as np
import matplotlib.pyplot as plt
#%%

def f( x ):
    return 1.0 / ( 1.0 + x )

def referenceFunction():
    return np.log( 2.0 )
#%%
def lowerSumFunction(y,h):
    
    integral = 0.0
   
    for i in range( 0, len(y)-1, 1 ):
        integral += y[i] * h
        
    return integral
#%%
def upperSumFunction(y,h):
    integral = 0.0
    
    return integral
#%%
def trapezoidal(y,h):
    
    integral = 0.0
    
    for i in range(0, len(y)-1, 1):
        integral += 0.5 * h * ( y[i] + y[i+1] )
    
    return integral
#%%

#N = 10
N = 100 # intervals
#N = 10000
#N = 1000000
#N = 100000000

x = np.linspace( 0.0, 1.0, N+1 )
h = x[1] - x[0]

y = f(x)

refValue = referenceFunction()

numericalValue = lowerSumFunction( y, h )
#numericalValue = trapezoidal(y, h)

errorValue = numericalValue - refValue
rel = errorValue / refValue

plt.figure(figsize=(10,8))
print('Ref =%1.6e Num = %1.6e Err = %1.6e' % ( refValue, numericalValue, errorValue ) )
plt.loglog( h, rel, 'bo' )
#plt.loglog(h, numericalValue, 'r+')

N = 10000

x = np.linspace( 0.0, 1.0, N+1 )
h = x[1] - x[0]

y = f(x)

refValue = referenceFunction()

numericalValue = lowerSumFunction( y, h )
#numericalValue = trapezoidal(y, h)

errorValue = numericalValue - refValue
rel = errorValue / refValue

print('Ref =%1.6e Num = %1.6e Err = %1.6e' % ( refValue, numericalValue, errorValue ) )
plt.loglog( h, rel, 'rs' )

N = 1000000
#N = 100000000

x = np.linspace( 0.0, 1.0, N+1 )
h =x[1] - x[0]

y = f(x)

refValue = referenceFunction()

numericalValue = lowerSumFunction( y, h )
#numericalValue = trapezoidal(y, h)

errorValue = numericalValue-refValue
rel = errorValue / refValue

print('Ref =%1.6e Num = %1.6e Err = %1.6e' % ( refValue, numericalValue, errorValue ) )
plt.loglog( h, rel, 'gd' )

## Expectation 
hval = np.linspace(1.e-8, 1.e-2, 100)
eps1 = 0.35 * hval
#plt.plot(hval,eps1, 'k--')
plt.loglog( hval, eps1, 'k--')


## Trapezoid rule
eps2 = 0.1 * hval**2
#plt.plot( hval, eps2,'k:' )
plt.loglog( hval, eps2,'k:' )

plt.show()

#%%


    