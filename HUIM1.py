# -*- coding: utf-8 -*-
"""
Created on Thu Aug 15 00:27:47 2024

@author: Gherbi
"""

import numpy as np
from commun import bmin
from itertools import combinations
'''
# Parameters
num_food_sources = 50  # Number of food sources (rows)
D = 30  # Number of dimensions (columns)

# Generate the food_sources array
food_sources = np.random.uniform(-100, 100, (num_food_sources, D))

# Copy a specific food source to new_solution
i = 5  # You can choose any index (0 to num_food_sources-1)
new_solution = np.copy(food_sources[i])
'''

# Convert new_solution into a transaction, including item positions
def create_single_transaction(solution):
    items = [f'{i}' for i in range(len(solution))]
    utilities = np.abs(solution)
    positions = [(0, i) for i in range(len(solution))]
    return {'items': items, 'utilities': utilities, 'positions': positions}



def generate_candidates(itemsets, length):
    return list(combinations(itemsets, length))

def calculate_utility_and_positions(itemset, transaction):
    utility = 0
    positions = []
    items = transaction['items']
    utilities = transaction['utilities']
    if all(item in items for item in itemset):
        indices = [items.index(item) for item in itemset]
        utility += sum(utilities[i] for i in indices)
        positions.extend([transaction['positions'][i] for i in indices])
    return utility, positions

def high_utility_itemset_mining(new_solution):
    #min_utility=0
    min_utility=bmin
    transaction = create_single_transaction(new_solution)
    frequent_itemsets = []
    
    # Generate initial itemsets of size 1
    items = transaction['items']
    candidates = generate_candidates(items, 1)
    
    while candidates:
        new_candidates = []
        for candidate in candidates:
            utility, positions = calculate_utility_and_positions(candidate, transaction)
            if utility >= min_utility:
                frequent_itemsets.append((candidate, utility, positions))
                new_candidates.extend(generate_candidates(candidate, 2))
        
        candidates = list(set(new_candidates))
    
    return frequent_itemsets
"""
# Apply HUIM algorithm on the single transaction
min_utility = 50  # Set a minimum utility threshold
frequent_itemsets = high_utility_itemset_mining(new_solution)

# Sort the frequent itemsets by utility in descending order
frequent_itemsets = sorted(frequent_itemsets, key=lambda x: x[1], reverse=True)
gbest=int(frequent_itemsets[0][0][0])
print('frequent_itemsets:',len(frequent_itemsets))
L=[]
print("Top 5 High-Utility Itemsets with Positions in new_solution:")
for itemset, utility, positions in frequent_itemsets[:5]:
    print(f"Itemset: {itemset}, Utility: {utility}")
    print("itemset:",itemset[0])
    L.append(utility)
    #for position in positions:
     #   print(f"   Position in new_solution array: Row {position[0]}, Column {position[1]}")
print("frequent_itemsets[0][0][0]:",frequent_itemsets[0][0][0])
print('frequent:',frequent_itemsets)
"""
