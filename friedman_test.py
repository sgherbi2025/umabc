# -*- coding: utf-8 -*-
"""
Created on Fri Aug 23 14:17:30 2024

@author: Gherbi
"""

import scipy.stats as stats

# Exemple de données
data = [
    [15, 18, 12],
    [20, 22, 19],
    [13, 17, 14],
    [25, 29, 27],
    [19, 21, 20]
]

# Appliquer le test de Friedman
stat, p_value = stats.friedmanchisquare(*data)

print(f"Statistique de test: {stat}, p-value: {p_value}")
