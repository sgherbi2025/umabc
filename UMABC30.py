# -*- coding: utf-8 -*-
"""
Created on Wed Aug 14 18:29:41 2024

@author: Gherbi
"""

import numpy as np
import random
import pandas as pd
import HUIM1  #algorithm applied on numerical data to find high_utility_itemset_mining
import benchmark26

"""
# Objective function (Sphere function as an example)
def objective_function(x):
    return np.sum(x**2)
"""


# Parameters
D = 30
limit = 50
MaxNFE = 5000 * D
num_runs = 30 

#Parameters functions
shift_vector = np.random.uniform(-100, 100, D)  # Random shift vector
rotation_matrix = np.random.randn(D, D)  # Random rotation matrix

# For F19, Schwefel's Problem 2.6 with bounds, define A and b accordingly
A = np.random.randn(D, D)
b = np.random.uniform(-100, 100, D)


def objective_function(x):
    #return sum(xi ** 2 for xi in x)
    return benchmark26.rosenbrock(x)

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
    food_sources = np.random.uniform(-30, 30, (num_food_sources, D))
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
                      
            # Apply HUIM algorithm
            #min_utility = 20  # Set a minimum utility threshold
            frequent_itemsets = HUIM1.high_utility_itemset_mining(new_solution)  
            # Sort the frequent itemsets by utility in descending order
            frequent_itemsets = sorted(frequent_itemsets, key=lambda x: x[1], reverse=True)
            
            # Print the top five best utility itemsets            
            L=[]            
            for itemset, utility, positions in frequent_itemsets[:5]:                          
                L.append(int(itemset[0]))            
            if len(L)!=0: k=random.choice(L)
            #print("L:",L)
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
        # Onlooker bees phase
        '''
        # Ensure all fitness values are non-negative
        epsilon=1e-8
        fitness = np.maximum(fitness, 0)+epsilon
        
        # Check for NaN in fitnesses
        if np.isnan(fitness).any():
            raise ValueError("NaN detected in fitness values!")
        
        # Calculate total fitness
        total_fitness = np.sum(fitness)
        
        # Guard against division by zero or very small sums
        if total_fitness == 0 or np.isnan(total_fitness):
            raise ValueError("Total fitness is zero or NaN, leading to invalid probability computation!")
        
        # Normalize fitnesses to get probabilities
        prob = fitness / total_fitness
        
        # Check for NaN in probabilities
        if np.isnan(prob).any():
            raise ValueError(f"NaN detected in probabilities! Fitnesses: {fitness}, Total Fitness: {total_fitness}")
    
        # Adjust probabilities to ensure they sum to 1
        prob_sum = np.sum(prob)
        
        if not np.isclose(prob_sum, 1.0):
            prob /= prob_sum  # Normalize again to ensure the sum is exactly 1
        
        # Ensure probabilities sum to 1
        if np.isnan(np.sum(prob)):
            raise ValueError(f"Probabilities sum to NaN after normalization! Sum: {np.sum(prob)}")
        
        assert np.isclose(np.sum(prob), 1.0), f"Probabilities do not sum to 1, sum is {np.sum(prob)}"
        '''
        
        prob = fitness / np.sum(fitness)
        for _ in range(num_food_sources):
            i = np.random.choice(np.arange(num_food_sources), p=prob)
            k = np.random.randint(0, D)
            phi = np.random.uniform(-1, 1)
            new_solution = np.copy(food_sources[i])  
            # Apply HUIM algorithm
            #min_utility = 500  # Set a minimum utility threshold
            frequent_itemsets = HUIM1.high_utility_itemset_mining(new_solution) 
            # Sort the frequent itemsets by utility in descending order
            frequent_itemsets = sorted(frequent_itemsets, key=lambda x: x[1], reverse=True)
            
            # The top five best utility itemsets            
            L=[]            
            for itemset, utility, positions in frequent_itemsets[:5]:                            
                L.append(int(itemset[0]))
            
            if len(L)!=0: k=random.choice(L) 
            new_solution[k] = food_sources[i][k] + phi * (food_sources[i][k] - food_sources[np.random.randint(0, num_food_sources)][k])
            new_fitness =calculate_fitness(objective_function(new_solution))
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
                #print('scout')
                #r1=np.random.uniform(-1, 1)
                #r2=r1-1
                # Apply HUIM algorithm
                #min_utility = 20  # Set a minimum utility threshold
               
                frequent_itemsets = HUIM1.high_utility_itemset_mining(new_solution)          
                
                r2 = np.random.uniform(0, 0.5)

                # Calculate r1 such that r1 + 2 * r2 = 1
                r1 = 1 - 2 * r2  
                r3=r2
                
                # Sort the frequent itemsets by utility in descending order
                frequent_itemsets = sorted(frequent_itemsets, key=lambda x: x[1], reverse=True)
                if len(frequent_itemsets)!=0: butility=int(frequent_itemsets[0][0][0])
                else: 
                    r1=np.random.uniform(-1, 1)
                    r3=1-r1
                    r2=0
                food_sources[i] = r1*np.random.uniform(-30,30, D)+r2*food_sources[butility]+r3*np.array(food_sources[i])
                
                #food_sources[i] = np.random.uniform(-100, 100, D)
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
    
        best_index = np.argmin(fitness)
        best_solution = food_sources[best_index]
        best_fitness = fitness[best_index]
        convergence.append(best_fitness)  
    # Best solution found
    best_index = np.argmin(fitness)
    best_solution = food_sources[best_index]
    best_fitness = fitness[best_index]
    
    return best_solution, best_fitness



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