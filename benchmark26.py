# -*- coding: utf-8 -*-
"""
Created on Mon Jul 15 21:35:14 2024

@author: Gherbi
"""

import numpy as np
import pandas as pd
from mlxtend.frequent_patterns import apriori, association_rules

# Define benchmark functions
def sphere(x):
    return sum(xi ** 2 for xi in x)

def schwefel2_22(x):
    return sum(xi * np.sin(np.sqrt(abs(xi))) for xi in x)

def schwefel1_2(x):
    return sum(abs(xi) + np.prod(abs(xi)) for xi in x)

def schwefel2_21(x):
    return sum([abs(xi) for xi in x])

def rosenbrock(x):
    return sum(100 * (xi1 - xi0**2)**2 + (1 - xi0)**2 for xi0, xi1 in zip(x[:-1], x[1:]))

def step(x):
    return sum(np.floor(xi + 0.5)**2 for xi in x)

def quartic_with_noise(x):
    return sum(i * xi**4 for i, xi in enumerate(x, start=1)) + np.random.normal(0, 1)

def schwefel2_26(x):
    return sum([xi * np.sin(np.sqrt(abs(xi))) for xi in x]) 

def rastrigin(x):
    A = 10
    return A * len(x) + sum(xi**2 - A * np.cos(2 * np.pi * xi) for xi in x)

def ackley(x):
    a = 20
    b = 0.2
    c = 2 * np.pi
    sum1 = sum(xi**2 for xi in x)
    sum2 = sum(np.cos(c * xi) for xi in x)
    return -a * np.exp(-b * np.sqrt(sum1 / len(x))) - np.exp(sum2 / len(x)) + a + np.exp(1)

def griewank(x):
    sum1 = sum(xi**2 for xi in x) / 4000
    prod = np.prod([np.cos(xi / np.sqrt(i + 1)) for i, xi in enumerate(x)])
    return sum1 - prod + 1

def penalized1(x):
    penalty = 1000 if any(abs(xi) > 50 for xi in x) else 0
    return sum(xi**2 for xi in x) + penalty

def penalized2(x):
    penalty = 1000 if any(abs(xi) > 50 for xi in x) else 0
    return sum(xi**2 for xi in x) + penalty

#Shifted unimodal functions

def shifted_sphere_function(x, shift_vector):
    return np.sum((x - shift_vector) ** 2)

def shifted_schwefel_problem_1_2(x, shift_vector):
    return np.sum([np.sum(x[:i+1] - shift_vector[:i+1]) ** 2 for i in range(len(x))])

def shifted_rotated_high_conditioned_elliptic(x, shift_vector, rotation_matrix):
    z = np.dot(rotation_matrix, x - shift_vector)
    return np.sum([10**6 ** (i / (len(x) - 1)) * z_i ** 2 for i, z_i in enumerate(z)])

def shifted_schwefel_problem_1_2_with_noise(x, shift_vector):
    base = shifted_schwefel_problem_1_2(x, shift_vector)
    noise = 1 + 0.4 * abs(np.random.normal(0, 1))
    return base * noise

def schwefel_problem_2_6_with_bounds(x, A, b):
    return np.max(np.abs(np.dot(A, x) - b))



# Rotation matrix generator
def generate_rotation_matrix(D):
    R = np.random.randn(D, D)
    Q, _ = np.linalg.qr(R)
    return Q

# Shifted Rosenbrock's function
def shifted_rosenbrock(x, shift_vector):
    z = x - shift_vector
    return sum(100 * (z[1:] - z[:-1]**2)**2 + (z[:-1] - 1)**2)

# Shifted Rotated Griewank's function without bounds
def shifted_rotated_griewank(x, shift_vector, rotation_matrix):
    z = np.dot(rotation_matrix, x - shift_vector)
    return np.sum(z**2 / 4000) - np.prod(np.cos(z / np.sqrt(np.arange(1, len(z) + 1)))) + 1

# Shifted Rotated Ackley's function with global optimum on bounds
def shifted_rotated_ackley(x, shift_vector, rotation_matrix):
    z = np.dot(rotation_matrix, x - shift_vector)
    return -20 * np.exp(-0.2 * np.sqrt(np.sum(z**2) / len(z))) - np.exp(np.sum(np.cos(2 * np.pi * z)) / len(z)) + 20 + np.e

# Shifted Rastrigin's function
def shifted_rastrigin(x, shift_vector):
    z = x - shift_vector
    return 10 * len(z) + np.sum(z**2 - 10 * np.cos(2 * np.pi * z))

# Shifted Rotated Rastrigin's function
def shifted_rotated_rastrigin(x, shift_vector, rotation_matrix):
    z = np.dot(rotation_matrix, x - shift_vector)
    return 10 * len(z) + np.sum(z**2 - 10 * np.cos(2 * np.pi * z))

# Shifted Rotated Weierstrass function
def shifted_rotated_weierstrass(x, shift_vector, rotation_matrix, a=0.5, b=3):
    z = np.dot(rotation_matrix, x - shift_vector)
    k_max = 20
    term1 = np.sum([np.sum([a**k * np.cos(2 * np.pi * b**k * (z_i + 0.5)) for k in range(k_max + 1)]) for z_i in z])
    term2 = len(z) * np.sum([a**k * np.cos(2 * np.pi * b**k * 0.5) for k in range(k_max + 1)])
    return term1 - term2

# Schwefel's Problem 2.13
def schwefel_problem_2_13(x):
    return np.sum(x * np.sin(np.sqrt(np.abs(x))))


'''
# Example usage:
D = 30  # Dimensionality of the problem
shift_vector = np.random.uniform(-100, 100, D)  # Random shift vector
rotation_matrix = np.random.randn(D, D)  # Random rotation matrix
x = np.random.uniform(-100, 100, D)

# Calculate each function
f15 = shifted_sphere_function(x, shift_vector)
f16 = shifted_schwefel_problem_1_2(x, shift_vector)
f17 = shifted_rotated_high_conditioned_elliptic(x, shift_vector, rotation_matrix)
f18 = shifted_schwefel_problem_1_2_with_noise(x, shift_vector)

# For F19, Schwefel's Problem 2.6 with bounds, define A and b accordingly
A = np.random.randn(D, D)
b = np.random.uniform(-100, 100, D)
f19 = schwefel_problem_2_6_with_bounds(x, A, b)

print(f"F15 (Shifted Sphere): {f15}")
print(f"F16 (Shifted Schwefel's Problem 1.2): {f16}")
print(f"F17 (Shifted Rotated High Conditioned Elliptic): {f17}")
print(f"F18 (Shifted Schwefel's Problem 1.2 with Noise): {f18}")
print(f"F19 (Schwefel's Problem 2.6 with Bounds): {f19}")
'''

