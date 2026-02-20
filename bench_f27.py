# -*- coding: utf-8 -*-
"""
Created on Thu Aug 15 21:25:17 2024

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

def shifted_rotated_expanded_scaffers_f6(x, shift_vector, rotation_matrix):
    """
    Calculates the value of the Shifted Rotated Expanded Scaffer's F6 function.
    
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
    for i in range(len(z) - 1):
        z_i = z[i]
        z_i1 = z[i + 1]
        term = 0.5 + (np.sin(np.sqrt(z_i**2 + z_i1**2))**2 - 0.5) / (1 + 0.001 * (z_i**2 + z_i1**2))**2
        result += term
    
    return result
"""
# Example usage
D = 30  # Dimensionality of the problem
shift_vector = np.random.uniform(-100, 100, D)  # Random shift vector
rotation_matrix = generate_rotation_matrix(D)  # Random rotation matrix
x = np.random.uniform(-100, 100, D)  # Random input vector

# Calculate the function value
f27_value = shifted_rotated_expanded_scaffers_f6(x, shift_vector, rotation_matrix)
print(f"Shifted Rotated Expanded Scaffer's F6 Function Value: {f27_value}")
"""