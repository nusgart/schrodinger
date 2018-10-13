#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 12 23:07:05 2018

@author: nnusgart
"""

###
import numpy as np
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import

import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import itertools
#import pandas as pd
#import seaborn as sns
from numba import jit

###############
### constants##
###############
scale = 1e-10
# hbar
hbar = 1.0545718e-34 # Js
# energy of the electron???
E = 1.0 # energy
# electron mass
mass = 9.10938356e-31 # kilograms
# electron charge
ech = 1.6e-19
#norm
a = hbar ** 2 / (mass * ech * ech)
## simulation parameters
N = 100
hfN = int(N/2)
tmax = 500
delta = 1
####
psi = np.zeros([N,N,N], dtype=np.complex64)
t = 0

############
### Functions
############
## potential energy
@jit(nopython=True)
def V(x, y, z):
    #if(x == hfN and y == hfN and z == hfN):
    #    return 0
    ech = 1.6e-19
    xx = (x - hfN) ** 2
    yy = (y - hfN) ** 2
    zz = (z - hfN) ** 2
    rr = (xx + yy + zz ) ** .5 * scale
    if (rr < 5 * scale):
        rr = 5 * scale
    return - ech*ech / rr

@jit(nopython=True)
def laplacePsi(x,y, z, h):
    ### X
    if (x >= h and x + h < N):
        px = psi[x+h,y,z] + psi[x-h, y, z]
    elif (x >= h):
        px = psi[x - h, y, z] + psi[x, y, z]
    elif (x + h < N):
        px = psi[x + h, y, z] + psi[x, y, z]
    else:
        px = psi[x, y, z] * 2
    ### Y
    if (y >= h and y + h < N):
        py = psi[x,y + h,z] + psi[x, y - h, z]
    elif (y >= h):
        py = psi[x, y - h, z] + psi[x, y, z]
    elif (y + h < N):
        py = psi[x, y + h, z] + psi[x, y, z]
    else:
        py = psi[x, y, z] * 2
    ### Z
    if (z >= h and z + h < N):
        pz = psi[x,y,z + h] + psi[x, y, z - h]
    elif (z >= h):
        pz = psi[x, y, z - h] + psi[x, y, z]
    elif (z + h < N):
        pz = psi[x, y, z + h] + psi[x, y, z]
    else:
        pz = psi[x, y, z] * 2
    ###
    return (px + py + pz - 6 * psi[x,y,z]) / h**2
@jit(nopython=True)
def dPsiDt(x, y, z):
    return -1j * hbar / (2 * mass) * laplacePsi(x,y,z,1) + 1j  / hbar * V(x,y,z)

## update the wave function using the schrodinger equation
@jit(nopython=True)
def updatePsi(psi, dt):
    for x in range(0, N):
        for y in range(0, N):
            for z in range(0, N):
                psi[x,y,z] += dt * dPsiDt(x, y, z)

## initialize psi
def initPsi():
    ### initialize psi
    norm = a**1.5 / np.pi ** .5 
    for x,y,z in itertools.product(*map(range, (N,N,N))):
        xx = (x - hfN) * scale
        yy = (y - hfN) * scale
        zz = (z - hfN) * scale
        r = np.sqrt(xx**2.0 + yy**2.0 + zz**2.0)
        psi[x,y,z] = norm * np.exp(-r/a)

def plotPsi():
    if int(t) % 10 == 0:
        print(t, " wavefcn ", psi)

print("Schrödinger Simulation of Hydrogen N =", N, " tmax =", tmax)
initPsi()
print("Starting Simulation")
while t < tmax:
    updatePsi(psi, delta)
    plotPsi()
    t += delta

print("Finished Simulating.")
print("Now writing out final data...")
np.savez_compressed("schrödinger.dat", psi)
print("Plotting data....")
fig = plt.figure()
ax = fig.gca(projection='3d')

# Make data.
X = np.arange(0, 100, 1)
Y = np.arange(0, 100, 1)
X, Y = np.meshgrid(X, Y)
ZZ = psi[X, Y, 50]
Z = np.real(ZZ * np.conjugate(ZZ))

# Plot the surface.
surf = ax.plot_surface(X, Y, Z, cmap=cm.coolwarm,
                       linewidth=0, antialiased=False)

# Customize the z axis.
#ax.set_zlim(-1.01, 1.01)
ax.zaxis.set_major_locator(LinearLocator(10))
ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))

# Add a color bar which maps values to colors.
fig.colorbar(surf, shrink=0.5, aspect=5)

plt.show()
fig.save("Schrödinger.png")
print("Done")