# -*- coding: utf-8 -*-
"""
Created on Wed Aug 14 18:29:41 2024

@author: Gherbi
"""

import numpy as np
import benchmark26

import bench_f26

# Parameters
D = 30
limit = 50
MaxNFE = 5000 * D
num_runs = 30 

#Parameters functions
shift_vector = np.random.uniform(-100,100, D)  # Random shift vector
rotation_matrix = np.random.randn(D, D)  # Random rotation matrix

# For F19, Schwefel's Problem 2.6 with bounds, define A and b accordingly
A = np.random.randn(D, D)
b = np.random.uniform(-100, 100, D)


# Objective function (Sphere function as an example)
def objective_function(x):
    return benchmark26.sphere(x)

def calculate_fitness(objective_value, epsilon=1e-8):
    # Fitness function ensures non-negative values
    if objective_value>=0:
        return 1.0 / (1.0 + objective_value + epsilon)
    else: return 1+abs(objective_value)
# ABC algorithm
def ABC_algorithm(D, limit, MaxNFE):
    # Parameters
    num_food_sources = D
    max_trials = limit
    MaxNFE = MaxNFE
    
    # Initialize food sources (solutions) and their fitness
    food_sources = np.random.uniform(-30,30, (num_food_sources, D))
    #fitness = np.array([objective_function(food_sources[i]) for i in range(num_food_sources)])
    fitness = [calculate_fitness(objective_function(sol)) for sol in food_sources]
    
    # Trials for each food source
    trials = np.zeros(num_food_sources)
    
    # Counter for the number of function evaluations
    NFE = num_food_sources
    convergence=[]
    # Main loop
    while NFE < MaxNFE:
        # Employed bees phase
        for i in range(num_food_sources):
            k = np.random.randint(0, D)
            phi = np.random.uniform(-1, 1)
            new_solution = np.copy(food_sources[i])
            new_solution[k] = food_sources[i][k] + phi * (food_sources[i][k] - food_sources[np.random.randint(0, num_food_sources)][k])
            new_fitness = calculate_fitness(objective_function(new_solution))
            NFE += 1
            
            # Greedy selection
            if new_fitness < fitness[i]:
                food_sources[i] = new_solution
                fitness[i] = new_fitness
                trials[i] = 0
            else:
                trials[i] += 1
            best_index = np.argmin(fitness)
            best_solution = food_sources[best_index]
            best_fitness = fitness[best_index]
            convergence.append(best_fitness)
        # Onlooker bees phase
        
        prob = fitness / np.sum(fitness)
        for _ in range(num_food_sources):
            i = np.random.choice(np.arange(num_food_sources), p=prob)
            k = np.random.randint(0, D)
            phi = np.random.uniform(-1, 1)
            new_solution = np.copy(food_sources[i])
            new_solution[k] = food_sources[i][k] + phi * (food_sources[i][k] - food_sources[np.random.randint(0, num_food_sources)][k])
            new_fitness = objective_function(new_solution)
            NFE += 1
            
            # Greedy selection
            if new_fitness < fitness[i]:
                food_sources[i] = new_solution
                fitness[i] = new_fitness
                trials[i] = 0
            else:
                trials[i] += 1
            best_index = np.argmin(fitness)
            best_solution = food_sources[best_index]
            best_fitness = fitness[best_index]
            convergence.append(best_fitness)
        # Scout bees phase
        for i in range(num_food_sources):
            if trials[i] > max_trials:
                food_sources[i] = np.random.uniform(-30,30, D)
                fitness[i] = objective_function(food_sources[i])
                trials[i] = 0
                NFE += 1
                if NFE >= MaxNFE:
                    break
                best_index = np.argmin(fitness)
                best_solution = food_sources[best_index]
                best_fitness = fitness[best_index]
                convergence.append(best_fitness)
        # Termination check
        if NFE >= MaxNFE:
            break
        num_food_sources
    # Best solution found
    best_index = np.argmin(fitness)
    best_solution = food_sources[best_index]
    best_fitness = fitness[best_index]
    
    return best_solution, best_fitness,convergence

#print(f"Mean Function Error: {mean_error:.2E} ± {std_deviation:.2E}")
"""
# Main execution
results = []

for run in range(num_runs):
    best_solution, best_fitness = ABC_algorithm(D, limit, MaxNFE)
    results.append(best_fitness)

# Calculate mean and standard deviation
mean_error = np.mean(results)
std_deviation = np.std(results)

# Print results in scientific notation with 2 decimal places
print(f"Mean Function Error: {mean_error:.2E} ± {std_deviation:.2E}")
"""
