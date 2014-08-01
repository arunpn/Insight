__author__ = 'brandonkelly'

import numpy as np
import matplotlib.pyplot as plt
import cPickle
import pandas as pd
import os
from sklearn.linear_model import MultiTaskLasso

base_dir = os.environ['HOME'] + '/Projects/Insight/'
data_dir = base_dir + 'data/yummly/'
plot_dir = base_dir + 'plots/'


def logit(x):
    return np.log(x / (1.0 - x))


def inv_logit(x):
    return np.exp(x) / (1.0 + np.exp(x))

flavors = pd.read_hdf(data_dir + 'pulled_pork_flavors.h5', 'df')
flavors = flavors.dropna()

search0 = cPickle.load(open(data_dir + 'pulled_pork_search.pickle', 'rb'))

# find ids of recipes that have flavor information
ids = []
for match in search0.matches:
    ids.append(match.id)

matches = []
for id in flavors.index:
    if id in ids:
        matches.append(search0.matches[ids.index(id)])

ingredients = []
for match in matches:
    ingredients.extend(match.ingredients)

unique_ingredients = np.unique(ingredients)
print 'Found', len(unique_ingredients), 'unique ingredients out of', len(flavors), 'recipes.'

X = np.zeros((len(flavors), len(unique_ingredients)))  # matrix of predictors
for i, match in enumerate(matches):
    for j, ingredient in enumerate(unique_ingredients):
        if ingredient in match.ingredients:
            X[i, j] = 1.0  # the i^{th} recipe contains the j^{th} ingredient

fvalues = flavors.values
fvalues[fvalues <= 0.001] = 0.01
fvalues[fvalues >= 0.999] = 0.99
y = logit(flavors.values)
lasso = MultiTaskLasso(alpha=1e-3).fit(X, y)

print 'Shape of lasso coefficients is', lasso.coef_.shape

for j, un_igred in enumerate(unique_ingredients):
    print un_igred, inv_logit(lasso.coef_[:, j])