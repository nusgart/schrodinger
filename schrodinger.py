#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 12 23:07:05 2018

@author: nnusgart
"""

###
import numpy as np
#import matplotlib.pyplot as plt
#import pandas as pd
#import seaborn as sns

###############
### constants##
###############
# hbar
hbar = 1.0545718e-34 # Js
# energy of the electron???
E = 1.0 # energy
# electron mass
mass = 9.10938356e-31 # kilograms
# electron charge
ech = 1.6e-19
#norm
a = 1.0
## simulation parameters
N = 100
tmax = 1000
delta = .01
####
psi = np.zeros([N,N,N], dtype=np.complex128)
t = 0

############
### Functions
############
## potential energy
def V(x, y, z):
    return - ech*ech / np.sqrt(x*x+y*y+z*z)

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

def dPsiDt(x, y, z):
    return -1j * hbar / (2 * mass) * laplacePsi(x,y,z,1) + 1j  / hbar * V(x,y,z)

## update the wave function using the schrodinger equation
def updatePsi(dt):
    for x in range(0, N):
        for y in range(0, N):
            for z in range(0, N):
                psi[x,y,z] += dt * dPsiDt(x, y, z)

## initialize psi
def initPsi():
    ### initialize psi
    for x in range(0, N):
        for y in range(0, N):
            for z in range(0, N):
                r = np.sqrt(x**2.0 + y**2.0 + z**2.0)
                psi[x,y,z] = a**1.5 / np.pi ** .5 * np.exp(-r/a)

def plotPsi():
    print(psi)

initPsi()
while t < tmax:
    updatePsi(delta)
    plotPsi()
    t += delta