# -*- coding: utf-8 -*-
"""
Created on Tue Aug 27 22:06:09 2024

@author: Gherbi
"""

import numpy as np
import matplotlib.pyplot as plt

# Sphere function definition
def sphere_function(x):
    return np.sum(x ** 2)

# Standard ABC Algorithm
def abc_standard(dim, bounds, max_iters, population_size):
    return abc_algorithm(dim, bounds, max_iters, population_size, variant="standard")

# Variant 1: Modified Search Phase
def abc_variant1(dim, bounds, max_iters, population_size):
    return abc_algorithm(dim, bounds, max_iters, population_size, variant="variant1")

# Variant 2: Modified Scout Phase
def abc_variant2(dim, bounds, max_iters, population_size):
    return abc_algorithm(dim, bounds, max_iters, population_size, variant="variant2")

# Generalized ABC Algorithm with variants
def abc_algorithm(dim, bounds, max_iters, population_size, variant="standard"):
    population = np.random.uniform(bounds[0], bounds[1], (population_size, dim))
    fitness = np.array([sphere_function(ind) for ind in population])
    
    best_fitness_over_time = []
    
    for iteration in range(max_iters):
        for i in range(population_size):
            # Generate a neighbor based on the variant
            if variant == "standard":
                neighbor = population[i] + np.random.uniform(-1, 1, dim)
            elif variant == "variant1":
                neighbor = population[i] + np.random.uniform(-2, 2, dim)  # More aggressive search
            elif variant == "variant2":
                neighbor = population[i] + np.random.normal(0, 1, dim)    # Gaussian mutation
            neighbor = np.clip(neighbor, bounds[0], bounds[1])
            neighbor_fitness = sphere_function(neighbor)
            
            if neighbor_fitness < fitness[i]:
                population[i] = neighbor
                fitness[i] = neighbor_fitness
        
        for i in range(population_size):
            selected_index = np.random.choice(np.arange(population_size))
            neighbor = population[selected_index] + np.random.uniform(-1, 1, dim)
            neighbor = np.clip(neighbor, bounds[0], bounds[1])
            neighbor_fitness = sphere_function(neighbor)
            
            if neighbor_fitness < fitness[selected_index]:
                population[selected_index] = neighbor
                fitness[selected_index] = neighbor_fitness
        
        worst_index = np.argmax(fitness)
        if np.random.uniform() < 0.1:
            if variant == "variant2":
                # In this variant, scout phase is more aggressive
                population[worst_index] = np.random.uniform(bounds[0], bounds[1], dim)
            else:
                population[worst_index] = np.random.uniform(bounds[0], bounds[1], dim)
            fitness[worst_index] = sphere_function(population[worst_index])
        
        best_fitness_over_time.append(np.min(fitness))
    
    return best_fitness_over_time

# Parameters
dim = 5
bounds = (-5.12, 5.12)
max_iters = 100
population_size = 30

# Run the ABC variants
standard_fitness = abc_standard(dim, bounds, max_iters, population_size)
variant1_fitness = abc_variant1(dim, bounds, max_iters, population_size)
variant2_fitness = abc_variant2(dim, bounds, max_iters, population_size)

# Plotting the convergence graphs
plt.plot(standard_fitness, label='Standard ABC')
plt.plot(variant1_fitness, label='ABC Variant 1')
plt.plot(variant2_fitness, label='ABC Variant 2')
plt.xlabel('Iterations')
plt.ylabel('Best Fitness')
plt.title('Convergence of ABC Variants on Sphere Function')
plt.legend()
plt.grid(True)
plt.show()
