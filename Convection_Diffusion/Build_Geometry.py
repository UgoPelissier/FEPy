#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Aug  6 14:18:15 2022

@author: Ugo PELISSIER
"""

import numpy as np
import matplotlib.pyplot as plt

def box(left_bottom, l, h):
    x = left_bottom[0]
    y = left_bottom[1]
    out = [left_bottom,
            [x+l,y],
            [x+l,y+h],
            [x,y+h],
            left_bottom]
    return out
    
def circle(center,r):
    theta = np.linspace(0, 2*np.pi, 11)
    x1 = r*np.cos(theta)+center[0]
    x2 = r*np.sin(theta)+center[1]
    circle = [[x1[i],x2[i]] for i in range(len(theta))]
    return circle

def circles(r, center):
    c = [circle(r[i], center[j]) for j in range(len(center)) for i in range(len(r))]
    return c

def plot_geometry(geometry):
    x,y = zip(*geometry)
    plt.plot(x, y, 'k', linewidth=1)
    plt.plot(x, y, 'ok',markersize=1)
    
def plot_geometries(geometry_list):
    for geometry in geometry_list:
        plot_geometry(geometry)