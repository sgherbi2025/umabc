# -*- coding: utf-8 -*-
"""
Created on Thu Aug 15 21:51:03 2024

@author: Gherbi
"""

import numpy as np

def generate_rotation_matrix(D):
    """
    Generates a random orthogonal rotation matrix.
    """
    R = np.random.randn(D, D)
    Q, _ = np.linalg.qr(R)
    return Q

def shifted_expanded_griewank_rosenbrock(x, shift_vector, rotation_matrix):
    """
    Calculates the value of the Shifted Expanded Griewank’s plus Rosenbrock’s Function (F8F2).
    
    Parameters:
    x : array-like
        Input vector.
    shift_vector : array-like
        Shift vector.
    rotation_matrix : array-like
        Rotation matrix.
        
    Returns:
    float
        The function value.
    """
    # Apply rotation and shift
    z = np.dot(rotation_matrix, x - shift_vector)
    
    # Compute the function value
    result = 0
    n = len(z)
    for i in range(n - 1):
        term1 = (z[i + 1] - z[i]**2)**2 / 4000
        term2 = (z[i] - 1)**2
        term3 = np.cos((z[i + 1] - z[i]**2) / np.sqrt(i + 1))
        result += term1 + term2 - term3
    
    return result
"""
# Example usage
D = 30  # Dimensionality of the problem
shift_vector = np.random.uniform(-100, 100, D)  # Random shift vector
rotation_matrix = generate_rotation_matrix(D)  # Random rotation matrix
x = np.random.uniform(-100, 100, D)  # Random input vector

# Calculate the function value
f26_value = shifted_expanded_griewank_rosenbrock(x, shift_vector, rotation_matrix)
print(f"Shifted Expanded Griewank’s plus Rosenbrock’s Function (F8F2) Value: {f26_value}")
"""