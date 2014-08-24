__author__ = 'brandonkelly'

import numpy as np
import matplotlib.pyplot as plt
import cPickle
import pandas as pd
import os
from ingredient_mapping import IngredientMapping

base_dir = os.environ['HOME'] + '/Projects/Insight/'
data_dir = base_dir + 'data/yummly/'
plot_dir = base_dir + 'plots/'

search = cPickle.load(open(data_dir + 'sauce_search.pickle', 'rb'))
IngMap = cPickle.load(open(data_dir + 'ingredient_map_test.pickle', 'rb'))

ingredient_names = np.unique(IngMap.values())
data = np.zeros((len(search), len(ingredient_names)))
recipe_ids = []
for match in search:
    recipe_ids.append(match.id)

recipes = pd.DataFrame(data=data, columns=ingredient_names, index=recipe_ids)

for match in search:
    ingredients = IngMap.map_ingredients(match.ingredients)
    for ingredient in ingredients:
        recipes.set_value(match.id, ingredient, 1.0)

normalized_values = recipes.values / recipes.sum(axis=1).values[:, np.newaxis]

recipes = pd.DataFrame(data=normalized_values, columns=ingredient_names, index=recipe_ids)

recipes.to_hdf(data_dir + 'sauce_recipes.h5', 'df')