__author__ = 'brandonkelly'

import numpy as np
from sklearn.cross_validation import train_test_split
from ingredient_mapping import IngredientMapping
from ingredient_graph import IngredientGraph
import pymysql
from pymysql.err import MySQLError
import pandas as pd


def build_test_recipes(ids, ingredients):
    recipes = []
    current_id = ids[0]
    this_recipe = []
    for id, ingredient in zip(ids, ingredients):
        if id != current_id:
            # finished this recipe, save and move on to the next one
            current_id = id
            recipes.append(this_recipe)
            this_recipe = []
        this_recipe.append(ingredient)

    return recipes


def remove_ingredients(recipes, graph):
    new_recipes = []
    left_out_ingredients = []
    for ingredients in recipes:
        shuffled = np.random.permutation(ingredients)
        for i in xrange(len(shuffled)):
            if shuffled[i] in graph.ingredient_names:
                # found an ingredient that is in the graph, so make a new recipe from its omission
                shuffled_l = list(shuffled)
                left_out = shuffled_l.pop(i)
                new_recipes.append(shuffled_l)
                left_out_ingredients.append(left_out)

    return new_recipes, left_out_ingredients


def get_recommendations(input_ingredients, flavor, nrecommendations):
    conn = pymysql.connect('localhost', 'root', '', 'recipes', charset='utf8')
    cur = conn.cursor()
    potential_ingredients = []
    # get the list of potential ingredients to recommend
    cur.execute("SELECT Ingredient1 FROM Ingredient_Graph LIMIT 1")
    # need to select the first ingredient manually since it never appears as Ingredient2 in the MySQL table
    potential_ingredients.append(cur.fetchone()[0].lower())
    cur.execute("SELECT DISTINCT Ingredient2 FROM Ingredient_Graph")
    rows = cur.fetchall()
    for row in rows:
        potential_ingredients.append(row[0].lower())
    ingredient_pmi = pd.Series(np.zeros(len(potential_ingredients)), index=potential_ingredients)

    # compute total pairwise mutual information for the set of ingredients
    for ingredient in input_ingredients:
        sql_command = "SELECT * FROM Ingredient_Graph WHERE Ingredient1 = '" + \
                      ingredient + "' or Ingredient2 = '" + ingredient + "'"
        try:
            cur.execute(sql_command)
        except MySQLError:
            continue
        rows = cur.fetchall()
        if len(rows) > 0:
            # only use similarity info for ingredients used to train the ingredient graph
            for row in rows:
                ingredient1 = row[0].lower()
                ingredient2 = row[1].lower()
                pmi = row[2]
                if ingredient1 not in input_ingredients:  # don't recommend ingredients already in the recipe
                    ingredient_pmi[ingredient1] += pmi
                elif ingredient2 not in input_ingredients:
                    ingredient_pmi[ingredient2] += pmi

    ingredient_pmi.sort(ascending=False)
    ingredient_pmi = ingredient_pmi[ingredient_pmi > 0]  # don't recommend statistically independent ingredients

    # find ingredients with the right flavor type
    recommended_ingredients = []
    for ingredient in ingredient_pmi.index:
        sql_command = "SELECT Type from Ingredient_Flavors WHERE Ingredient = '" + ingredient + "'"
        cur.execute(sql_command)
        row = cur.fetchall()
        if len(row) == 0:
            type = "none"
        else:
            type = row[0][0].lower()
        if type != 'none':
            if flavor != "any":
                if type == flavor:
                    recommended_ingredients.append(ingredient)
            else:
                recommended_ingredients.append(ingredient)

    recommended_ingredients = recommended_ingredients[:nrecommendations]

    return recommended_ingredients

print "Loading data..."
graph = IngredientGraph(verbose=True, nprior=1e4)
graph.load_ingredient_map()
ids1, ingredients1 = graph.load_ingredients(table='Ingredient_List_Graph')
ids2, ingredients2 = graph.load_ingredients()  # ingredient labels as sauces or condiments
ids2 = list(np.array(ids2) + np.max(ids1) + 1)
ids1.extend(ids2)
ingredients1.extend(ingredients2)

# train / test split
print 'Doing train/test split...'
ids = np.array(ids1)
ingredients = np.array(ingredients1)

unique_ids = np.unique(ids)
uids_train, uids_test = train_test_split(unique_ids)
ids = np.array(ids)
ingredients = np.array(ingredients)
ing_train = []
ids_train = []
for recipe_id in uids_train:
    ids_train.extend(ids[ids == recipe_id])
    ing_train.extend(ingredients[ids == recipe_id])
ing_test = []
ids_test = []
for recipe_id in uids_test:
    ids_test.extend(ids[ids == recipe_id])
    ing_test.extend(ingredients[ids == recipe_id])

# train the graph
X_train = graph.build_design_matrix(ids_train, ing_train, min_counts=50)
print 'Learning graph...'
graph.fit(X_train)

# build the test set
print 'Building the test recipes...'
recipes = build_test_recipes(ids_test, ing_test)
print 'Building the test set...'
test_recipes, left_out_ingredients = remove_ingredients(recipes, graph)

# now find fraction of times left-out ingredient is recommended in the top 25% vs a random recommendation
nrecommended = 0
nrandom = 0
print 'Analyzing test set of', len(test_recipes), 'recipes...'
for i in xrange(len(test_recipes)):
    print i
    recommended = get_recommendations(test_recipes[i], 'any', 25)
    random = np.random.permutation(graph.ingredient_names)[0]
    if left_out_ingredients[i] in recommended:
        nrecommended += 1
    if left_out_ingredients[i] in random:
        nrecommended += 1

print 'Recommended left out ingredient in top 25', nrecommended, 'times out of', len(test_recipes), \
    '(' + str(100.0 * float(nrecommended) / len(test_recipes)) + '%)'
print 'Recommended left out ingredient in top 25', nrandom, 'times out of', len(test_recipes), \
    ' for random recommendations (' + str(100.0 * float(nrandom) / len(test_recipes)) + '%)'
