# -*- coding: utf-8 -*-
"""
Created on Wed Aug 14 22:08:19 2024

@author: Gherbi
"""


from mlxtend.frequent_patterns import fpgrowth
from mlxtend.preprocessing import TransactionEncoder
import pandas as pd

# Example dataset (list of transactions)
dataset = [
    ['milk', 'bread', 'butter'],
    ['bread', 'butter'],
    ['milk', 'bread'],
    ['milk', 'butter'],
    ['bread', 'butter', 'milk', 'cheese'],
    ['bread', 'milk']
]

# Convert the dataset into a one-hot encoded DataFrame
te = TransactionEncoder()
te_ary = te.fit(dataset).transform(dataset)
df = pd.DataFrame(te_ary, columns=te.columns_)
print('df:',df)
# Apply FP-Growth algorithm
frequent_itemsets = fpgrowth(df, min_support=0.5, use_colnames=True)

# Display the results
print(frequent_itemsets)
