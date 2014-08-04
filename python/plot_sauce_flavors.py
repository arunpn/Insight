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

flav_names = ['salty', 'sour', 'sweet', 'bitter', 'meaty', 'piquant']

search = cPickle.load(open(data_dir + 'sauce_search.pickle', 'rb'))
IngMap = cPickle.load(open(data_dir + 'ingredient_map_test.pickle', 'rb'))
# find the distribution of flavors for this recipe
flavors = []
ids = []
for match in search:
    flav_list = []
    ids.append(match.id)
    for flavor in flav_names:
        flav_list.append(match.flavors[flavor])
    flavors.append(flav_list)

flavors = np.vstack(flavors).astype(float)
flavors = pd.DataFrame(data=flavors, columns=flav_names, index=ids)

flavors.to_hdf(data_dir + 'sauce_flavors.h5', 'df')

ax = flavors.hist(bins=7)
for i in range(ax.shape[0]):
    for j in range(ax.shape[1]):
        ax[i, j].set_xlim(0, 1)
plt.savefig(plot_dir + 'sauce_flavors.png')
plt.close()

ingredients = []
for match in search:
    ingredients.extend(match.ingredients)

ingredients = IngMap.map_ingredients(ingredients)

unique_ingredients = np.unique(ingredients)
print 'Found', len(unique_ingredients), 'unique ingredients out of', len(flavors), 'recipes.'
icounts = []
for uing in unique_ingredients:
    icounts.append(ingredients.count(uing))

sorted = np.argsort(icounts)
icounts = np.array(icounts)

icounts = icounts[sorted][::-1]
unique_ingredients = unique_ingredients[sorted][::-1]

# first plot without ingredient labels
pos = np.arange(len(unique_ingredients)) + 0.5
plt.plot(pos, icounts, lw=3)
plt.ylabel('Counts (out of ' + str(len(flavors)) + ' recipes)')
plt.xlabel('Sorted Ingredient Index')
plt.savefig(plot_dir + 'sauce_ingred_counts.png')
plt.close()

# now plot but with ingredient labels, but only plot top 100 because of crowding
icounts = icounts[:29]
unique_ingredients = unique_ingredients[:29]

pos = np.arange(len(unique_ingredients)) + 0.5
plt.barh(pos, icounts, align='center')
plt.xlabel('Counts (out of ' + str(len(flavors)) + ' recipes)')
plt.yticks(pos, unique_ingredients)
plt.tight_layout()
plt.savefig(plot_dir + 'sauce_ingred_counts_top.png')
plt.close()