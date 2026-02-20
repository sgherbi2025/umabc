# -*- coding: utf-8 -*-
"""
Created on Wed Aug 14 18:29:41 2024

@author: Gherbi
"""

import numpy as np
import matplotlib.pyplot as plt
import ABC30
import UMABC30

"""
# ABC Main execution
UMABCresults = []

for run in range(UMABC30.num_runs):
    best_solution, best_fitness = UMABC30.ABC_algorithm(UMABC30.D,UMABC30.limit, UMABC30.MaxNFE)
    UMABCresults.append(best_fitness)

# Calculate mean and standard deviation
mean_error = np.mean(UMABCresults)
std_deviation = np.std(UMABCresults)

# Print results in scientific notation with 2 decimal places
print(f"UMABC Mean Function Error: {mean_error:.2E} ± {std_deviation:.2E}")

# UMABC Main execution
ABCresults = []
for run in range(ABC30.num_runs):
    best_solution, best_fitness = ABC30.ABC_algorithm(ABC30.D, ABC30.limit, ABC30.MaxNFE)
    ABCresults.append(best_fitness)

# Calculate mean and standard deviation
mean_error = np.mean(ABCresults)
std_deviation = np.std(ABCresults)

# Print results in scientific notation with 2 decimal places
print(f"ABC Mean Function Error: {mean_error:.2E} ± {std_deviation:.2E}")
"""
best_solution, best_fitness,ABCconv = ABC30.ABC_algorithm(ABC30.D, ABC30.limit, ABC30.MaxNFE)
UMbest_solution, UMbest_fitness,UMABCconv = ABC30.ABC_algorithm(UMABC30.D, UMABC30.limit, UMABC30.MaxNFE)
# Plotting the convergence graphs
plt.plot(ABCconv, label='ABC')
plt.plot(UMABCconv, label='UMABC')
plt.xlabel('NFE')
plt.ylabel('Error value')
plt.title('Convergence of ABC and UMABC on f6')
plt.legend()
plt.grid(True)
plt.show()