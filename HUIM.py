# -*- coding: utf-8 -*-
"""
Created on Wed Aug 14 22:37:52 2024

@author: Gherbi
"""

import numpy as np
import pandas as pd
from itertools import combinations

# Parameters
num_food_sources = 30  # Number of food sources (rows)
D = 30  # Number of dimensions (columns)

# Generate the data array
data = np.random.uniform(-100, 100, (num_food_sources, D))
print('data:',data)
print('data:',len(data))
# Convert data into transactions
def create_transactions(data):
    transactions = []
    for row in data:
        items = [f'{i}' for i in range(len(row))]
        utilities = np.abs(row)
        transactions.append({'items': items, 'utilities': utilities})
    return transactions

transactions = create_transactions(data)

def get_item_utilities(transactions):
    item_utilities = {}
    for transaction in transactions:
        items = transaction['items']
        utilities = transaction['utilities']
        for item, utility in zip(items, utilities):
            if item in item_utilities:
                item_utilities[item] += utility
            else:
                item_utilities[item] = utility
    return item_utilities

def generate_candidates(itemsets, length):
    return list(combinations(itemsets, length))

def calculate_utility(itemset, transactions):
    utility = 0
    for transaction in transactions:
        items = transaction['items']
        utilities = transaction['utilities']
        if all(item in items for item in itemset):
            indices = [items.index(item) for item in itemset]
            utility += sum(utilities[i] for i in indices)
    return utility

def high_utility_itemset_mining(transactions, min_utility):
    item_utilities = get_item_utilities(transactions)
    frequent_itemsets = []
    
    # Generate initial itemsets of size 1
    items = list(item_utilities.keys())
    candidates = generate_candidates(items, 1)
    
    while candidates:
        new_candidates = []
        for candidate in candidates:
            utility = calculate_utility(candidate, transactions)
            if utility >= min_utility:
                frequent_itemsets.append((candidate, utility))
                new_candidates.extend(generate_candidates(candidate, 2))
        
        candidates = list(set(new_candidates))
    
    return frequent_itemsets

# Apply HUIM algorithm
min_utility = 500  # Set a minimum utility threshold
frequent_itemsets = high_utility_itemset_mining(transactions, min_utility)

# Display results
print("High-Utility Itemsets:")
for itemset, utility in frequent_itemsets:
    print(f"Itemset: {itemset}, Utility: {utility}")


# Sort the frequent itemsets by utility in descending order
frequent_itemsets = sorted(frequent_itemsets, key=lambda x: x[1], reverse=True)

# Print the top five best utility itemsets
print("Top 5 High-Utility Itemsets:")
L=[]
for itemset, utility in frequent_itemsets[:5]:
    print(f"Itemset: {int(itemset[0])}, Utility: {utility}")
    L.append(int(itemset[0]))
print('list:',L)    