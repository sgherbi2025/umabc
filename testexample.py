# -*- coding: utf-8 -*-
"""
Created on Wed Aug 28 14:57:35 2024

@author: Gherbi
"""
import numpy as np

# Objective Function (Sphere Function)
def objective_function(x):
    return np.sum(x**2)

# ABC Algorithm
def abc_algorithm(objective_function, dim, bounds, n_bees=50, n_iter=100):
    # Initialize parameters
    lower_bound, upper_bound = bounds
    bees = np.random.uniform(lower_bound, upper_bound, (n_bees, dim))
    fitness = np.array([objective_function(bees[i]) for i in range(n_bees)])
    best_bee_idx = np.argmin(fitness)
    best_solution = bees[best_bee_idx]
    best_cost = fitness[best_bee_idx]
    
    # ABC Algorithm Main Loop
    for _ in range(n_iter):
        # Employed Bees Phase
        for i in range(n_bees):
            k = np.random.randint(0, n_bees)
            while k == i:
                k = np.random.randint(0, n_bees)
            phi = np.random.uniform(-1, 1, dim)
            new_bee = bees[i] + phi * (bees[i] - bees[k])
            new_bee = np.clip(new_bee, lower_bound, upper_bound)
            new_cost = objective_function(new_bee)
            if new_cost < fitness[i]:
                bees[i] = new_bee
                fitness[i] = new_cost
                if new_cost < best_cost:
                    best_solution = new_bee
                    best_cost = new_cost

        # Onlooker Bees Phase
        prob = (fitness - np.min(fitness)) / (np.sum(fitness) - n_bees * np.min(fitness))
        prob = np.clip(prob, 1e-10, 1)  # Avoid division by zero and zero probabilities
        for i in range(n_bees):
            if np.random.rand() < prob[i]:
                k = np.random.randint(0, n_bees)
                while k == i:
                    k = np.random.randint(0, n_bees)
                phi = np.random.uniform(-1, 1, dim)
                new_bee = bees[i] + phi * (bees[i] - bees[k])
                new_bee = np.clip(new_bee, lower_bound, upper_bound)
                new_cost = objective_function(new_bee)
                if new_cost < fitness[i]:
                    bees[i] = new_bee
                    fitness[i] = new_cost
                    if new_cost < best_cost:
                        best_solution = new_bee
                        best_cost = new_cost

        # Scout Bees Phase
        best_fitness = np.min(fitness)
        for i in range(n_bees):
            if np.random.rand() < (fitness[i] - best_fitness) / (np.sum(fitness) - n_bees * best_fitness):
                bees[i] = np.random.uniform(lower_bound, upper_bound, dim)
                fitness[i] = objective_function(bees[i])
                if fitness[i] < best_cost:
                    best_solution = bees[i]
                    best_cost = fitness[i]

    return best_solution, best_cost

# Main function to run multiple experiments and calculate statistics
def run_experiments(n_runs, dim, bounds, n_bees, n_iter):
    true_optimal_value = 0  # True value for the objective function at the optimal solution
    errors = []

    for _ in range(n_runs):
        _, best_cost = abc_algorithm(objective_function, dim, bounds, n_bees, n_iter)
        error = abs(best_cost - true_optimal_value)
        errors.append(error)

    mean_error = np.mean(errors)
    std_deviation = np.std(errors)
    
    return mean_error, std_deviation

# Example Usage
if __name__ == "__main__":
    dim = 10  # Dimension of the problem
    bounds = (-10, 10)  # Bounds for the variables
    n_bees = 50  # Number of bees in the swarm
    n_iter = 100  # Number of iterations
    n_runs = 20  # Number of runs to calculate mean error and std deviation

    mean_error, std_deviation = run_experiments(n_runs, dim, bounds, n_bees, n_iter)

    print(f"Mean Error: {mean_error}")
    print(f"Standard Deviation: {std_deviation}")
