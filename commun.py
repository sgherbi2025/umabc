# -*- coding: utf-8 -*-
"""
Created on Sun Sep  1 14:20:36 2024

@author: Gherbi
"""
import bench_f27
import numpy as np
import benchcec2013
import benchmark26
bmin=-100
bmax=100
global_optimum=-450
D=30
#Parameters functions
shift_vector = np.random.uniform(bmin, bmax, D)  # Random shift vector
rotation_matrix = np.random.randn(D, D)  # Random rotation matrix

# For F19, Schwefel's Problem 2.6 with bounds, define A and b accordingly
A = np.random.randn(D, D)
b = np.random.uniform(bmin, bmax, D)

def objective_function(x):
    #return sum(xi ** 2 for xi in x)
    return benchmark26.shifted_sphere_function(x, shift_vector)