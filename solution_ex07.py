#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul  4 00:27:41 2024

@author: tusker
"""

import numpy as np
import matplotlib.pyplot as plt
#%%

# INITITALIZATION
n = 0.0
t = 0.0

x = -8.0
y = -1.0
z = 33.0

dt = 1.0e-3
t_end = 40.0

# control params
b = 8.0 / 3.0
r = 28.0
s = 10.0

x_arr = [x]
y_arr = [y]
z_arr = [z]
t_arr = [t]
#%%
# lorenz derivative

while( t < t_end ):
    ## UPDATE THE NEW = OLD + UPDATE
    x_new = x + dt * ( s * ( y - x ) )
    y_new = y + dt * ( (( r - z ) * x) - y )
    z_new = z + dt * ( (x * y) - ( b * z ) )
    
    x = x_new
    y = y_new
    z = z_new
    t += dt
    n += 1
    
    # data storage
    x_arr.append(x)
    y_arr.append(y)
    z_arr.append(z)
    
    t_arr.append(t)

    # print output
    print('t= ',t, 'x=',x)

print('tEnd = ', t, ' xEnd', x)

x_arr = np.array( x_arr )
y_arr = np.array( y_arr )
z_arr = np.array( z_arr )
t_arr = np.array( t_arr )

#%%

# mean function
def compute_mean( values ):
    return sum( values ) / len( values )

# Compute the mean values for x, y, and z
mean_x = compute_mean( x_arr )
mean_y = compute_mean( y_arr )
mean_z = compute_mean( z_arr )

# Print the mean values
print( 'Mean of x: %1.6f' % mean_x)
print( 'Mean of y: %1.6f' % mean_y)
print( 'Mean of x: %1.6f' % mean_z)

#%%

# standard deviation function
def compute_sd( values, mean ):
    variance = sum( ( x - mean ) **2 for x in values ) / len( values )
    return np.sqrt( variance )

# Compute the standard deviation values for x, y, and z
std_x = compute_sd( x_arr, mean_x)
std_y = compute_sd( y_arr, mean_y)
std_z = compute_sd( z_arr, mean_z)

print( 'Standard deviation of x: %1.6f' % std_x)
print( 'Standard deviation of y: %1.6f' % std_y)
print( 'Standard deviation of z: %1.6f' % std_z)

#%%
# Post-processing
fig = plt.figure(figsize=(18, 6))

ax1 = fig.add_subplot( 131 )
ax1.plot( t_arr, x_arr, 'b' )
ax1.plot( [ 0, t_end], [ mean_x, mean_x ], 'k--', alpha=0.5)
ax1.plot( t_arr[ np.argmin( np.abs( x_arr - mean_x ) ) ], mean_x, 'ko')
ax1.set_title('x vs t')
ax1.set_xlabel('Time')
ax1.set_ylabel('x')

ax2 = fig.add_subplot( 132 )
ax2.plot( t_arr, y_arr, 'g' )
ax2.plot([0, t_end], [ mean_y, mean_y ], 'k--', alpha=0.5)
ax2.plot( t_arr[ np.argmin( np.abs( y_arr - mean_y ) ) ], mean_y, 'ko' )
ax2.set_title( 'y vs t' )
ax2.set_xlabel( 'Time' )
ax2.set_ylabel( 'y' )


ax3 = fig.add_subplot(133)
ax3.plot( t_arr, z_arr, 'r')
ax3.plot( [ 0, t_end ], [ mean_z, mean_z ], 'k--', alpha=0.5 )
ax3.plot( t_arr[ np.argmin( np.abs( z_arr - mean_z ) ) ], mean_z,'ko' )
ax3.set_title( 'z vs t' )
ax3.set_xlabel( 'Time' )
ax3.set_ylabel( 'z' )

plt.tight_layout()
plt.show()

#%%
# PLOT HISTOGRAMS 

fig2, axs = plt.subplots( 1, 3, figsize=( 18, 6 ) )

axs[0].hist( x_arr, bins=200, density=True, color='b', alpha=0.7)
axs[0].set_title('Histogram and PDF of x')
axs[0].set_xlabel('x')
axs[0].set_ylabel('Density')

axs[1].hist( y_arr, bins=200, density=True, color='g', alpha=0.7)
axs[1].set_title('Histogram and PDF of y')
axs[1].set_xlabel('y')
axs[1].set_ylabel('Density')

axs[2].hist( z_arr, bins=200, density=True, color='r', alpha=0.7)
axs[2].set_title('Histogram and PDF of y')
axs[2].set_xlabel('z')
axs[2].set_ylabel('Density')

plt.tight_layout()
plt.show()

#%%
#2-D histograms JPDF

x_plus_y = x_arr + y_arr
x_minus_y = x_arr - y_arr

fig3, axs = plt.subplots( 1, 2, figsize=( 12, 6 ) )

# (x + y) vs (x - y) histogram and JPDF
hist = axs[0].hist2d( x_plus_y, x_minus_y, bins=40, density=False, cmap='Blues' )
axs[0].set_title( '2D Histogram of x + y and x - y' )
axs[0].set_xlabel( 'x + y' )
axs[0].set_ylabel( 'x - y' )
plt.colorbar( hist[3], ax=axs[0] )

# (x + y) vs (x - y) JPDF
jpdf = axs[1].hist2d( x_plus_y, x_minus_y, bins=40, density=True, cmap='Reds' )
axs[1].set_title( 'JPDF of x + y and x - y' )
axs[1].set_xlabel( 'x + y' )
axs[1].set_ylabel( 'x - y' )
plt.colorbar( jpdf[3], ax=axs[1] )

plt.tight_layout()
plt.show()
