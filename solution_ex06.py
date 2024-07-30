#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 25 12:07:17 2024

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
# TIME LOOP
while( t < t_end ):
    ## UPDATE THE NEW = OLD + UPDATE
    x_new = x + dt * ( s * ( y - x ) )
    y_new = y + dt * ( ( ( r - z ) * x ) - y )
    z_new = z + dt * ( ( x * y ) - ( b * z ) )
    
    x = x_new
    y = y_new
    z = z_new
    t += dt
    n += 1
    
    # data storage
    x_arr.append( x )
    y_arr.append( y )
    z_arr.append( z )
    
    t_arr.append( t )

    # print output
    print('t= ',t, 'x=',x)

print('tEnd = ', t, ' xEnd', x)    

#%%
## POST PROCESS -> PLOTS, DATAIO
# Store the file
np.savetxt( 'xdata.dat', x_arr )
np.savetxt( 'ydata.dat', y_arr )
np.savetxt( 'zdata.dat', z_arr )
np.savetxt( 'tdata.dat', t_arr )

#load the file
xb = np.loadtxt( 'xdata.dat' )
yb = np.loadtxt( 'ydata.dat' )
zb = np.loadtxt( 'zdata.dat' )
tb = np.loadtxt( 'tdata.dat' )

# make the list to arrays
xa = np.array( x_arr )
ya = np.array( y_arr )
za = np.array( z_arr )
ta = np.array( t_arr )

#%%

# Plotting x vs. z
plt.figure( figsize=(18, 6) )
plt.subplot(1, 3, 1)
plt.plot(xa, za, 'r-')
plt.xlabel('X')
plt.ylabel('Z')
plt.title('X vs. Z')

# Plotting y vs. z
plt.subplot(1, 3, 2)
plt.plot(ya, za, 'b-')
plt.xlabel('Y')
plt.ylabel('Z')
plt.title('Y vs. Z')

# Plotting x + y vs. x - y
plt.subplot(1, 3, 3)
plt.plot(xa + ya, xa - ya, 'g-')
plt.xlabel('X + Y')
plt.ylabel('X - Y')
plt.title('X + Y vs. X - Y')

plt.tight_layout()
plt.show()
#%%
# 3-D view
#from mpl_toolkits.mplot3d import Axes3D
fig3d = plt.figure( figsize=( 10, 8 ) )
ax3d = fig3d.add_subplot( 111, projection='3d' )

ax3d.view_init( azim = 105.0, elev = 28.0 )

ax3d.set_xlabel( 'x' )
ax3d.set_ylabel( 'y' )
ax3d.set_zlabel( 'z' )

# Plot the Lorenz attractor
ax3d.plot( xa, ya, za, 'b-' )
ax3d.plot( xb, yb, zb, 'r--' )

# 2-D view
fig2 = plt.figure( 2, figsize=( 10, 8 ) )
ax = fig2.add_subplot( 111 )

ax.plot( xa, ya,'b-' )
ax.plot( xb, yb,'r+' )

plt.show()