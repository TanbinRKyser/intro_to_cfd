#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr 14 21:15:54 2024

@author: tusker
"""

import numpy as np
import matplotlib.pyplot as plt
#%%
# Task 3(b)
x = np.zeros( 11 )

for i in range( 0, len(x), 1 ) :
    x[ i ] = i * 0.1

#print(x)
#%%
# Task 3(c)
start = 0 #-1
stop = 1 #1
step_size = 0.025 #step_size (dx)

num_elements = int( ( stop - start ) / step_size ) + 1
#print(num_elements)

x,step_size = np.linspace( start, stop, num_elements, retstep=True )
#print(x)

y1 = np.zeros_like( x )
y2 = np.zeros_like( x )
y3 = np.zeros_like( x )

# f(x)= 1-2x
y1 = 1 - 2*x

# f(x)=(x-0.4)^2
y2 = ( x - 0.4 )**2

# f(x)=sin(2*pi*x)
y3 = np.sin( 2 * np.pi * x )
#%%
# Task 4

plt.figure( figsize=( 10, 6 ) )
plt.title('Figure 1')
plt.xlabel('x')
plt.ylabel('y')

plt.xlim( 0, 1 )
plt.ylim( -1.2, 1.2 )

plt.plot( x, y1, 'o-', label='f1' )
plt.plot( x, y2, '--', label='f2' )
plt.plot( x, y3, '-.', label='f3' )

plt.grid()
plt.legend()

plt.show()
#%%
## Task 5

plt.figure(figsize=(10,6))
plt.title('Figure 2')

plt.xlim( 0, 0.5 )
plt.ylim( -1.2, 1.2 )

plt.plot( x, y3, '-', label='f3' )

plt.show()
