__author__ = 'brandonkelly'

import numpy as np
import matplotlib.pyplot as plt
import cPickle
import pandas as pd
import os
from ingredient_mapping import IngredientMapping
from scipy.optimize import nnls


base_dir = os.environ['HOME'] + '/Projects/Insight/'
data_dir = base_dir + 'data/yummly/'
plot_dir = base_dir + 'plots/'

flav_names = ['salty', 'sour', 'sweet', 'bitter', 'meaty', 'piquant']

flavors = pd.read_hdf(data_dir + 'sauce_flavors.h5', 'df')
recipes = pd.read_hdf(data_dir + 'sauce_recipes.h5', 'df')

y = flavors['piquant'].values
X = recipes.values

X = X[np.isfinite(y)]
y = y[np.isfinite(y)]

print 'Performing non-negative least-squares...'
weights, rnorm = nnls(X, y)
print 'Residual norm:', rnorm

plt.plot(X.dot(weights), y, '.', ms=2)
plt.plot(plt.xlim(), plt.xlim())
plt.xlabel('Estimated Sweetness')
plt.ylabel('True Sweetness')
plt.show()

counts = X > 0
counts = counts.sum(axis=0)
sorted_idx = np.argsort(counts)

for i in range(len(weights)):
    print recipes.columns[sorted_idx[i]], '  ', weights[sorted_idx[i]], counts[sorted_idx[i]]