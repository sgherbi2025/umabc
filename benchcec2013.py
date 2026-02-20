# -*- coding: utf-8 -*-
"""
Created on Thu Aug 29 15:01:00 2024

@author: Gherbi
"""

import numpy as np

# Sphere Function
def sphere_function(x):
    return np.sum(np.square(x))

# Rotated High Conditioned Elliptic Function
def rotated_high_conditioned_elliptic_function(x):
    return np.sum([10**6**(i/len(x)) * x[i]**2 for i in range(len(x))])

# Rotated Bent Cigar Function
def rotated_bent_cigar_function(x):
    return x[0]**2 + 10**6 * np.sum(np.square(x[1:]))

# Rotated Discus Function
def rotated_discus_function(x):
    return 10**6 * x[0]**2 + np.sum(np.square(x[1:]))

# Different Powers Function
def different_powers_function(x):
    return np.sum(np.abs(x)**(2 + 4 * np.arange(len(x)) / (len(x) - 1)))

# Rotated Rosenbrock’s Function
def rotated_rosenbrock_function(x):
    return np.sum(100 * (x[1:] - x[:-1]**2)**2 + (x[:-1] - 1)**2)

# Rotated Schaffers F7 Function
def rotated_schaffers_f7_function(x):
    return np.sum(np.sqrt(np.abs(x[:-1]**2 + x[1:]**2)) * (1 + np.sin(50 * np.sqrt(x[:-1]**2 + x[1:]**2))**2))

# Rotated Ackley’s Function
def rotated_ackley_function(x):
    return -20 * np.exp(-0.2 * np.sqrt(np.mean(np.square(x)))) - np.exp(np.mean(np.cos(2 * np.pi * x))) + 20 + np.e

# Rotated Weierstrass Function
def rotated_weierstrass_function(x, a=0.5, b=3, k_max=20):
    return np.sum([np.sum([a**k * np.cos(2 * np.pi * b**k * (x[i] + 0.5)) for k in range(k_max)]) for i in range(len(x))])

# Rotated Griewank’s Function
def rotated_griewank_function(x):
    return 1 + np.sum(np.square(x)) / 4000 - np.prod(np.cos(x / np.sqrt(np.arange(1, len(x) + 1))))

# Rastrigin’s Function
def rastrigin_function(x):
    return 10 * len(x) + np.sum(np.square(x) - 10 * np.cos(2 * np.pi * x))

# Rotated Rastrigin’s Function
def rotated_rastrigin_function(x):
    return rastrigin_function(x)  # Rotation can be added if needed

# Non-Continuous Rotated Rastrigin’s Function
def non_continuous_rotated_rastrigin_function(x):
    y = np.where(np.abs(x) > 0.5, np.round(2 * x) / 2, x)
    return rotated_rastrigin_function(y)

# Schwefel's Function
def schwefel_function(x):
    return 418.9829 * len(x) - np.sum(x * np.sin(np.sqrt(np.abs(x))))

# Rotated Schwefel's Function
def rotated_schwefel_function(x):
    return schwefel_function(x)  # Rotation can be added if needed

# Rotated Katsuura Function
def rotated_katsuura_function(x):
    product = 1
    for i in range(len(x)):
        sum_term = np.sum([abs(2**j * x[i] - np.floor(2**j * x[i])) / 2**j for j in range(1, 33)])
        product *= (1 + (i + 1) * sum_term)
    return product - 1

# Lunacek Bi-Rastrigin Function
def lunacek_bi_rastrigin_function(x, mu0=2.5, d=1.0):
    s = 1 - 1 / (2 * np.sqrt(len(x) + 20) - 8.2)
    return np.sum((x - mu0)**2) + 10 * len(x) - 10 * np.sum(np.cos(2 * np.pi * (x - mu0)))

# Rotated Lunacek Bi-Rastrigin Function
def rotated_lunacek_bi_rastrigin_function(x):
    return lunacek_bi_rastrigin_function(x)

# Expanded Griewank’s plus Rosenbrock’s Function
def expanded_griewank_rosenbrock_function(x):
    return np.sum(100 * (x[1:] - x[:-1]**2)**2 + (x[:-1] - 1)**2 / 4000 - np.prod(np.cos(x[:-1] / np.sqrt(np.arange(1, len(x[:-1]) + 1)))))

# Expanded Scaffer’s F6 Function
def expanded_schaffers_f6_function(x):
    return np.sum([0.5 + (np.sin(np.sqrt(x[i]**2 + x[i+1]**2))**2 - 0.5) / (1 + 0.001 * (x[i]**2 + x[i+1]**2))**2 for i in range(len(x)-1)])

# Composition Function 1 (n=5, Rotated)
def composition_function_1(x):
    # Placeholder example for a simple composition of 5 rotated functions
    f_list = [rotated_rastrigin_function, rotated_schwefel_function, rotated_ackley_function, sphere_function, rotated_griewank_function]
    weights = np.ones(5)
    return np.sum([weights[i] * f_list[i](x) for i in range(5)])

# Composition Function 2 (n=3, Unrotated)
def composition_function_2(x):
    f_list = [rastrigin_function, schwefel_function, ackley_function]
    weights = np.ones(3)
    return np.sum([weights[i] * f_list[i](x) for i in range(3)])

# Composition Function 3 (n=3, Rotated)
def composition_function_3(x):
    f_list = [rotated_rastrigin_function, rotated_schwefel_function, rotated_ackley_function]
    weights = np.ones(3)
    return np.sum([weights[i] * f_list[i](x) for i in range(3)])

# Composition Function 4 (n=3, Rotated)
def composition_function_4(x):
    return composition_function_3(x)  # Similar to Composition Function 3, adjust if needed

# Composition Function 5 (n=3, Rotated)
def composition_function_5(x):
    return composition_function_3(x)  # Similar to Composition Function 3, adjust if needed

# Composition Function 6 (n=5, Rotated)
def composition_function_6(x):
    return composition_function_1(x)  # Similar to Composition Function 1, adjust if needed

# Composition Function 7 (n=5, Rotated)
def composition_function_7(x):
    return composition_function_1(x)  # Similar to Composition Function 1, adjust if needed

# Composition Function 8 (n=5, Rotated)
def composition_function_8(x):
    return composition_function_1(x)  # Similar to Composition Function 1, adjust if needed

