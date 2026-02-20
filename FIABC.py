# -*- coding: utf-8 -*-
"""
Created on Wed Aug 14 18:29:41 2024

@author: Gherbi
"""

import numpy as np
import random
import pandas as pd
import HUIM1  #algorithm applied on numerical data to find high_utility_itemset_mining
from commun import global_optimum,bmin, bmax,objective_function
import time

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
"""
#Parameters functions
shift_vector = np.random.uniform(bmin, bmax, D)  # Random shift vector
rotation_matrix = np.random.randn(D, D)  # Random rotation matrix

# For F19, Schwefel's Problem 2.6 with bounds, define A and b accordingly
A = np.random.randn(D, D)
b = np.random.uniform(bmin, bmax, D)

def objective_function(x):
    #return sum(xi ** 2 for xi in x)
    return benchmark26.schwefel2_21(x)
"""

def calculate_fitness(objective_value, epsilon=0):
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
    food_sources = np.random.uniform(bmin, bmax, (num_food_sources, D))
    #fitness = np.array([objective_function(food_sources[i]) for i in range(num_food_sources)])
    fitness = [objective_function(sol) for sol in food_sources]
    
    # Trials for each food source
    trials = np.zeros(num_food_sources)
    
    # Counter for the number of function evaluations
    NFE = 0
    
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
        
        # Onlooker bees phase
        realfitness=[calculate_fitness(fit) for fit in fitness]
        prob = realfitness / np.sum(realfitness)
        #prob = fitness / np.sum(fitness)
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
            for itemset, utility, positions in frequent_itemsets[:10]:                            
                L.append(int(itemset[0]))
            
            if len(L)!=0: k=random.choice(L) 
            new_solution[k] = food_sources[i][k] + phi * (food_sources[i][k] - food_sources[np.random.randint(0, num_food_sources)][k])
            new_fitness =objective_function(new_solution)
            NFE += 1
            
            # Greedy selection
            if new_fitness < fitness[i]:
                food_sources[i] = new_solution
                fitness[i] = new_fitness
                trials[i] = 0
            else:
                trials[i] += 1
        
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
                food_sources[i] = r1*np.random.uniform(bmin, bmax, D)+r2*food_sources[butility]+r3*np.array(food_sources[i])
                
                #food_sources[i] = np.random.uniform(-100, 100, D)
                fitness[i] = objective_function(food_sources[i])
                trials[i] = 0
                NFE += 1
                
        # Termination check
        if NFE >= MaxNFE:
            break
    
    # Best solution found
    best_index = np.argmin(fitness)
    best_solution = food_sources[best_index]
    best_fitness = fitness[best_index]
    
    return best_solution, best_fitness




#Main execution
results = []
times=[]
for run in range(num_runs):
    tstart=time.time()
    best_solution, best_fitness = ABC_algorithm(D, limit, MaxNFE)
    tend=time.time()
    time_taken = tend - tstart
    results.append(abs(best_fitness-global_optimum))
    times.append(time_taken)

# Calculate mean and standard deviation
mean_error = np.mean(results)
std_deviation = np.std(results)

# Print results in scientific notation with 2 decimal places
print(f"Mean Function Error UMABC: {mean_error:.2E} ± {std_deviation:.2E}")

# Measure the time for one run
print(f"Time taken for one run UMABC: {np.mean(times):.2f} seconds")