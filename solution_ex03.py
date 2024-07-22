#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun  4 10:16:00 2024

@author: tusker
"""
import numpy as np
import matplotlib.pyplot as plt

from numpy import random as rnd
from scipy.interpolate import griddata
#%%
# Task 1(a):
    
A = np.array([ [11, 12, 13], 
              [21, 22, 23] ])
    
print(A)
#%%
# Task 1(b):
A[1, 2] = 99 

print(A)
#%%
# Task 1(c):

#print(A[4,4])
# index 4 is out of bounds for axis 0 with size 2
#%%
# Task 2(a):
Nx = 50
Ny = 30

x1d = np.linspace( 0.0, 4.0, Nx )
y1d = np.linspace( -1.0, 1.0, Ny )

x2d,y2d = np.meshgrid( x1d, y1d )
#%%
# Task 2(b): meshgrid
plt.figure(figsize=(10, 8))
plt.scatter( x2d, y2d )
plt.show()
#%%
# Task 2(c): take input
"""x_lower_limit = int( input("Please enter the lower limit of x: ") )
x_higher_limit = int( input("Please enter the higher limit of x: ") )
x_number_of_cells = int( input("Please enter the number of cells on the x axis: ") )
y_lower_limit = int( input("Please enter the lower limit of y: ") )
y_higher_limit = int( input("Please enter the higher limit of y: ") )
y_number_of_cells = int( input("Please enter the number of cells on the y axis: ") )

x = np.linspace( x_lower_limit, x_higher_limit, x_number_of_cells )
y = np.linspace( y_lower_limit, y_higher_limit, y_number_of_cells )

x2d,y2d = np.meshgrid( x, y )"""
#%%
# Task 2(d):
#np.savetxt('grid_x.csv', x2d, delimiter=',') 
np.savetxt('grid_x.dat', x2d, delimiter=',' ) # all X coordinates
#np.savetxt('grid_y.csv', y2d, delimiter=',') # all Y coordinates
np.savetxt('grid_y.dat', y2d, delimiter=',') 
#%%

#Task 3(a):
def stretching( xi, a = 0.99 ):

    b = 0.5 * np.log( ( 1.0+a ) / ( 1.0-a ) )
    
    #E = xi / 100
    
    stretched_y = np.tanh( b * ( xi - 0.5 ) ) / np.tanh( b / 2.0 )
    
    return stretched_y

# Generate a
xi = np.linspace( 0.0, 1.0, 30 )

y1d = stretching( xi, )
x1d = np.linspace( 0.0, 4.0, 50 )

x2d_stretch, y2d_stretch = np.meshgrid( x1d, y1d )

plt.figure(figsize=(10, 8))
plt.scatter( x2d_stretch, y2d_stretch, color="r" )
#plt.plot(x2d,y2d,'b')
#plt.show()

# Retrieve old data
x2d_equi = np.loadtxt('grid_x.dat', delimiter=',')
y2d_equi = np.loadtxt('grid_y.dat', delimiter=',')

plt.scatter( x2d_equi, y2d_equi, color='b', marker='+')
#plt.show()

#C-style row major storage
x2d = np.zeros((30,50))
y2d = np.zeros((30,50))

for j in range(30):
    for i in range(50):
        x2d[j,i] = x1d[i]
        y2d[j,i] = y1d[j]

plt.scatter( x2d, y2d, color='g', marker='s', s=2 )
plt.show()

#%%
#Task 4:

def data( x, y ):
    return np.sin( 2 * np.pi * x ) * np.cos( 8 * np.pi * y ) * np.exp( -4 * y**2)

unit = np.linspace( 0.0, 1.0, 200 )
x1d = y1d = unit

x2d,y2d = np.meshgrid( x1d, y1d )

z2d = data( x2d, y2d )

plt.figure( figsize=( 10, 8 ) )
plt.contourf( x2d, y2d, z2d, 256 )
#%%
# Task 4 (low resolution grid )
points = rnd.random_sample( ( 100, 2 ) )

x1d = points[ :, 0 ]
y1d = points[ :, 1 ]

z1d = data( x1d, y1d )

plt.figure( figsize=( 10, 8 ) )
plt.scatter( x1d, y1d, color='k', marker='o', zorder = 5 )

#z2d_interp = griddata( ( x1d, y1d, ), z1d, ( x2d, y2d ) )
#z2d_interp = griddata( ( x1d, y1d, ), z1d, ( x2d, y2d ), method='linear' )
z2d_interp = griddata( ( x1d, y1d, ), z1d, ( x2d, y2d ), method='nearest' )
#z2d_interp = griddata( ( x1d, y1d, ), z1d, ( x2d, y2d ), method='cubic' )

plt.contourf( x2d, y2d, z2d_interp, 256 )
#plt.scatter(x1d, y1d, marker='o', c='k', s=3)
#%%