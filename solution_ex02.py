#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun  3 11:14:43 2024

@author: tusker
"""
#%%
import numpy as np
import matplotlib.pyplot as pl
#%%
#Task 1(a):
    
N = 100
d = 2.0 * 10**( -4 )

def my_sum( N = 100, d = 2.0 * 10**( -4 ) ):
    
    result = 0    
    
    for n in range( 0, N+1, 1 ):
        result += n * d
        
    return result
#%%
# Task 1(b):
def ref_sum(N,d):
    
    return 0.5 * d * N * ( N+1 )
#%%
# Task 1(c):
"""def test_sum():
    mys=my_sum(N,d)
    refs=ref_sum(N,d)
    
    if mys == refs: pass
    else: raise ValueError
"""
print( repr( my_sum( N, d ) ) )
print( repr( ref_sum( N, d ) ) )
#%%
# Task 4(b):
def factorial( n ):
    if( n < 2 ): return 1
    else: return n * factorial( n-1 )
#%%
"""
import math

def factorial_test(n):
    nFactorial = factorial(n)
    nFactorial_ref = math.factorial(n)

    if( nFactorial == nFactorial_ref ):
        pass
    else:
        raise  
factorial_test(5)
"""
#%%    
""" Taylor series of exp(2x-1) around x_0=-1/2 up to 4th order"""

def Taylor4( xi ):
    
    yi = np.zeros_like( xi )
    
    for i in range( len( xi ) ):
        
        print('Taylor4 len(xi) = %i i=%i xi[i] = %1.6f' % ( len(xi), i, xi[i], ) )
        
        yi[i] = ( 1.0 - 2.0 * ( xi[i] + 0.5 )
                      + 4.0 * ( xi[i] + 0.5 )**2 / 2.0
                      - 8.0 * ( xi[i] + 0.5 )**3 / 6.0
                      +16.0 * ( xi[i] + 0.5 )**4 /24.0
                      )
    return yi

x = np.linspace( -1.0, 4.0, 101 )

yr = np.exp( -2.0 * x - 1.0 )
y4 = Taylor4( x )

pl.figure(figsize=(10,6))
pl.plot( -0.5, 1.0,'ko' )
pl.plot( x, yr,'b-', label='np.exp' )
pl.plot( x, y4,'r--', lw='2', label='Taylor4' )

pl.grid()
pl.ylim( [ -1.0, 3.0 ] )

pl.legend()

pl.show()
#%%

def Taylor_N( xi, N ):
    """
    Taylor series of exp(-2.0*x - 1.0) around x_0 = -1/2 up to arbitrary order N
    """    

    yi = np.empty_like(xi)
    sum_yi = np.zeros_like(xi)
    
    for n in range( 0, N+1, 1 ):
        
        print("TaylorN N = %i n = %i" % ( N, n,) )
        
        yi[:] = ( -2.0 *  ( xi[:] + 0.5 ) )**n / factorial( n )
        
        sum_yi[:] += yi[:]
        
    return sum_yi

# N = 2
N = 8
# N = 150
x = np.linspace( -1.0, 4.0, 101 )

yr = np.exp( -2.0*x - 1.0 )
y4 = Taylor4( x )
yN = Taylor_N( x, N )

pl.figure( figsize=( 10, 6 ) )
pl.plot( -0.5, 1.0, 'ko' )
pl.plot( x, yr, 'k-', label = 'np.exp')
pl.plot( x, y4, 'r--', label = 'Taylor4')
pl.plot( x, yN, 'b:', lw='2', label = 'TaylorN (N=%i)' % N )

pl.grid()
pl.ylim( [ -1.0, 3.0 ] )

pl.legend(loc='upper right')
pl.show()