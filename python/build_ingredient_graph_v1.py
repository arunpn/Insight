__author__ = 'brandonkelly'

import numpy as np
import matplotlib.pyplot as plt
import cPickle
import pandas as pd
import os
from ingredient_mapping import IngredientMapping
from scipy.optimize import nnls
from sklearn.covariance import GraphLassoCV
import pymysql


def mutual_information(recipe_ids, ingredients, ingredient1, ingredient2):
    recipes = np.unique(recipe_ids)
    ingredient1_idx = ingredients == ingredient1
    recipes_with1 = np.unique(recipe_ids[ingredient1_idx])
    ingredient2_idx = ingredients == ingredient2
    recipes_with2 = np.unique(recipe_ids[ingredient2_idx])

    recipes_with_both = np.intersect1d(recipes_with1, recipes_with2)

    frac1 = len(recipes_with1) / float(len(recipes))
    frac2 = len(recipes_with2) / float(len(recipes))
    frac12 = len(recipes_with_both) / float(len(recipes))

    pmi = np.log(frac12 / (frac1 * frac2))
    return pmi


base_dir = os.environ['HOME'] + '/Projects/Insight/'
data_dir = base_dir + 'data/yummly/'
plot_dir = base_dir + 'plots/'

# grab the ingredients for each recipe
conn = pymysql.connect('localhost', 'root', '', 'recipes')
cur = conn.cursor()
cur.execute("SELECT * FROM Ingredient_List")
rows = cur.fetchall()
recipe_ids = []
ingredients = []
for row in rows:
    recipe_ids.append(row[0])
    ingredients.append(row[1])

recipe_ids = np.array(recipe_ids)
ingredients = np.array(ingredients)

# construct the graph
uingredients = np.unique(ingredients)

graph = np.zeros((len(uingredients), len(uingredients)))

# first get number of recipes with each ingredient
nrecipes_single = np.zeros(len(uingredients))
nrecipes = len(np.unique(recipe_ids))
recipes = []
for i in xrange(len(uingredients)):
    print i
    ingredient_idx = ingredients == uingredients[i]
    these_recipes = np.unique(recipe_ids[ingredient_idx])
    nrecipes_single[i] = len(these_recipes)
    recipes.append(list(these_recipes))  # recipes is a list of recipe_id lists for each ingredient

# now get ingredient pairs
nrecipes_pairs = np.zeros((len(uingredients), len(uingredients)))
print 'Getting recipe pairs...'
for i, recipe_list in enumerate(recipes[:-1]):
    print i
    for j, other_recipe_list in enumerate(recipes[i+1:]):
        overlap = set(recipe_list).intersection(set(other_recipe_list))
        if len(overlap) > max(nrecipes_single[i], nrecipes_single[j]):
            print i, j, len(overlap), nrecipes_single[i], nrecipes_single[j]
        nrecipes_pairs[i, j] = len(overlap)

nrecipes_pairs = nrecipes_pairs + nrecipes_pairs.T

graph = np.zeros(nrecipes_pairs.shape)
for i in xrange(nrecipes_pairs.shape[0]):
    pair_prob = (nrecipes_pairs[i] + 1.0) / (nrecipes + 2.0)
    prob1 = (nrecipes_single[i] + 1.0) / (nrecipes + 2.0)
    prob2 = (nrecipes_single + 1.0) / (nrecipes + 2.0)
    graph[i] = np.log(pair_prob / (prob1 * prob2))  # pairwise mutual information

min_values = graph.min()
for i in xrange(graph.shape[0]):
    print i
    for j in xrange(graph.shape[0]):
        if nrecipes_single[i] < 10 or nrecipes_single[j] < 10:
            # only use ingredient that appear at least 10 times
            graph[i, j] = min_values
        if i == j:
            graph[i, j] = min_values / 2.0

graph = pd.DataFrame(graph, index=uingredients, columns=uingredients)
graph.to_hdf(data_dir + 'ingredient_graph.h5', 'df')