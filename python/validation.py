__author__ = 'brandonkelly'

import numpy as np
from sklearn.cross_validation import train_test_split
from ingredient_mapping import IngredientMapping
from ingredient_graph import IngredientGraph
import pymysql
from pymysql.err import MySQLError
import pandas as pd
import multiprocessing
import time

do_not_recommend = \
    [
        'angel hair',
        'arborio rice',
        'artichoke heart marin',
        'back ribs',
        'baguette',
        'baking mix',
        'baking powder',
        'baking soda',
        'basmati rice',
        'bass',
        'bean threads',
        'beef',
        'bertolli alfredo sauc',
        'bertolli tomato & basil sauc',
        'biscuits',
        'bisquick',
        'bread',
        'brisket',
        'brown rice',
        'bulk italian sausag',
        'buns',
        'cake',
        'catfish',
        'cheese',
        'cheese slices',
        'cheese soup',
        'chicken',
        'chips',
        'chuck',
        'cod',
        'cooked rice',
        'cooking spray',
        'corn husks',
        'corn starch',
        'cornflour',
        'cornish hens',
        'cornmeal',
        'cream of celery soup',
        'cream of chicken soup',
        'cutlet',
        'dough',
        'dressing',
        'duck',
        'dumpling wrappers',
        'egg noodles, cooked and drained',
        'egg roll wrappers',
        'egg roll wraps',
        'english muffins',
        'essence',
        'fat',
        'fettuccine pasta',
        'fettuccini',
        'filet',
        'filet mignon',
        'fillets',
        'fish',
        'flavoring',
        'florets',
        'flounder',
        'flour',
        'french baguett',
        'french onion soup',
        'frozen mix veget',
        'fryer chickens',
        'gelatin',
        'grating cheese',
        'grits',
        'ground meat',
        'ground round',
        'ground sausage',
        'halibut',
        'hamburger',
        'hominy',
        'hot dog bun',
        'hot dogs',
        'ice',
        'ice cream',
        'italian sauce',
        'jasmine rice',
        'juice',
        'kraft miracle whip dressing',
        'lamb',
        'lasagna noodles',
        'lasagna noodles, cooked and drained',
        'liquid',
        'loin',
        'long-grain rice',
        'macaroni',
        'mahi mahi',
        'manicotti',
        'margarine',
        'marinade',
        'meat',
        'meat tenderizer',
        'meatballs',
        'minute rice',
        'nonstick spray',
        'noodles',
        'oil',
        'orzo',
        'pasta',
        'pasta sauce',
        'pastry',
        'phyllo',
        'pie crust',
        'pie shell',
        'pitas',
        'pizza crust',
        'pizza doughs',
        'pizza sauce',
        'polenta',
        'pork',
        'potato chips',
        'potato starch',
        'prebaked pizza crusts',
        'pretzel stick',
        'processed cheese',
        'quail',
        'ragu pasta sauc',
        'red food coloring',
        'red snapper',
        'refrigerated piecrusts',
        'rib',
        'rice',
        'rice paper',
        'rice sticks',
        'roast',
        'roasting chickens',
        'rolls',
        'round steaks',
        'rub',
        'rump roast',
        'rump steak',
        'salad',
        'salmon',
        'salt',
        'sauce',
        'sausage casings',
        'seasoning',
        'shells',
        'short-grain rice',
        'shortening',
        'sirloin',
        'sole',
        'soup',
        'soup mix',
        'spaghetti, cook and drain',
        'spam',
        'spareribs',
        'spices',
        'spring roll wrappers',
        'steak',
        'steamed rice',
        'stew meat',
        'strip steaks',
        'stuffing mix',
        'sushi rice',
        'sweetener',
        'swordfish',
        'taco shells',
        'textured soy protein',
        'tilapia',
        'toast',
        'tomato soup',
        'tortilla chips',
        'tortillas',
        'trout',
        'tuna',
        'turkey',
        'udon',
        'veal',
        'vegetables',
        'veggies',
        'venison',
        'water',
        'wheat',
        'wheat bread',
        'white rice',
        'whitefish',
        'wide egg noodles',
        'wonton skins',
        'wonton wrappers',
        'yeast',
        'yellow corn meal'
    ]


def build_test_recipes(ids, ingredients):
    """
    Convert the input set of recipe IDs and ingredients into a list of lists, representing a list of recipes.

    :param ids: The recipe ID corresponding to each ingredient.
    :param ingredients: The list ingredients.
    :return: A list of lists containing the list of recipes.
    """
    recipes = []
    current_id = ids[0]
    this_recipe = []
    nrecipes = 0
    for id, ingredient in zip(ids, ingredients):
        if id != current_id:
            # finished this recipe, save and move on to the next one
            current_id = id
            recipes.append(this_recipe)
            this_recipe = [ingredient]
            nrecipes += 1
        else:
            this_recipe.append(ingredient)

    recipes.append(this_recipe)
    nrecipes += 1
    print len(recipes), nrecipes

    return recipes


def remove_ingredients(recipes, graph):
    """
    Create a new set of recipes by randomly removing one ingredient from each recipe.

    :param recipes: The list of recipes.
    :param graph: The Ingredient graph object, must contain the list of ingredient names.
    :return: A list of tuples containing the new recipe and the left-out ingredient.
    """
    divided_recipes = []
    for ingredients in recipes:
        shuffled = np.random.permutation(ingredients)
        for i in xrange(len(shuffled)):
            if shuffled[i] in graph.ingredient_names and shuffled[i] not in do_not_recommend:
                # found an ingredient that is in the graph and not blacklisted, so make a new recipe from its omission
                shuffled_l = list(shuffled)
                left_out = shuffled_l.pop(i)
                divided_recipes.append((shuffled_l, left_out))
                break

    return divided_recipes


def get_recommendations(input_ingredients, flavor, nrecommendations):
    """
    Compute the list of recommended ingredients by comparing the strengths of the graph edges between the input recipe's
    ingredients and the ingredients not in the recipe.

    :param input_ingredients: The list of ingredients input by the user, after being filtered by the ingredient map.
    :param flavor: The desired flavor profile: can be 'sweet', 'savory', 'piquant', or 'any'.
    :param nrecommendations: The number of recommended ingredients to return.
    :return: The list of recommended ingredients.
    """
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
                if ingredient1 not in input_ingredients and ingredient1 not in do_not_recommend:
                    ingredient_pmi[ingredient1] += pmi
                elif ingredient2 not in input_ingredients and ingredient2 not in do_not_recommend:
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


def recommend_random_ingredient(input_ingredients, graph, nrecommendations):
    """
    Recommend a set of randomly chosen ingredients to enhance a recipe.

    :param input_ingredients: The ingredients in the user-supplied recipe.
    :param graph: The IngradientGraph instance, needed for the list of ingredient names.
    :param nrecommendations: The number of ingredient to recommend.
    :return: A list of recommended ingredients.
    """
    random = np.random.permutation(graph.ingredient_names)
    recommended = []
    for ingredient in random:
        if ingredient not in input_ingredients and ingredient not in do_not_recommend:
            recommended.append(ingredient)
        if len(recommended) == nrecommendations:
            break

    return recommended


def recommend_common_ingredient(input_ingredients, graph, nrecommendations):
    """
    Recommend the most common ingredients to enhance a recipe.

    :param input_ingredients: The ingredients in the user-supplied recipe.
    :param graph: The IngradientGraph instance, needed for the list of ingredient names.
    :param nrecommendations: The number of ingredient to recommend.
    :return: A list of recommended ingredients, ordered by most commonly used.
    """
    sorted_idx = np.argsort(graph.train_marginal)[::-1]
    sorted_ingredients = np.array(graph.ingredient_names)[sorted_idx]
    recommended = []
    for ingredient in sorted_ingredients:
        if ingredient not in input_ingredients and ingredient not in do_not_recommend:
            recommended.append(ingredient)
        if len(recommended) == nrecommendations:
            break

    return recommended


def get_all_recommendations(args):
    """
    Check if the left-out ingredient is in the set of ingredients recommended using a trained IngredientGraph instance,
    recommneded at random, and by recommending the most common ingredients.

    :param args: A tuple containing the recipe and the left-out ingredient.
    :return: A tuple containing three boolean values. The boolean variables are true if the left-out ingredient is
        contained within the set of recommendations from the IngredientGraph, made at random, or among the most common
        ingredients, respectively.
    """
    test_recipe, left_out_ingredient = args
    recommended = get_recommendations(test_recipe, 'any', 5)
    random = recommend_random_ingredient(test_recipe, graph, 5)
    common = recommend_common_ingredient(test_recipe, graph, 5)
    print 'Recommended:', recommended
    print 'Random:', random
    print 'Common:', common
    recommended_graph = False
    recommended_random = False
    recommended_common = False
    if left_out_ingredient in recommended:
        recommended_graph = True
    if left_out_ingredient in random:
        recommended_random = True
    if left_out_ingredient in common:
        recommended_common = True

    return recommended_graph, recommended_random, recommended_common

t1 = time.clock()

print "Loading data..."
graph = IngredientGraph(verbose=True, nprior=1e4)
graph.load_ingredient_map()
ids1, ingredients1 = graph.load_ingredients(table='Ingredient_List_Graph')
ids2, ingredients2 = graph.load_ingredients()  # ingredient labels as sauces or condiments
ids2 = list(np.array(ids2) + np.max(ids1) + 1)
ids1.extend(ids2)
ingredients1.extend(ingredients2)

print 'Found', len(np.unique(ids1)), 'recipes and', len(np.unique(ingredients1)), 'ingredients.'

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

print 'Using', len(np.unique(ids_train)), 'unique recipes for the training set and', len(np.unique(ids_test)), \
    'unique recipes for the test set.'

# train the graph
X_train = graph.build_design_matrix(ids_train, ing_train, min_counts=5)
print 'Training X shape:', X_train.shape
print 'Learning graph...'
graph.fit(X_train)

# build the test set
print 'Building the test recipes...'
recipes = build_test_recipes(ids_test, ing_test)
print len(recipes)
print 'Building the test set...'
args = remove_ingredients(recipes, graph)

# now find fraction of times left-out ingredient is recommended in the top 25% vs a random recommendation
n_jobs = 1
print 'Analyzing test set of', len(args), 'recipes...'

pool = multiprocessing.Pool(n_jobs)
if n_jobs > 1:
    recommendations = pool.map(get_all_recommendations, args)
else:
    recommendations = map(get_all_recommendations, args)

recommendations = np.array(recommendations)
nrecommended = recommendations.sum(axis=0)

print 'Recommended left out ingredient in top 5', nrecommended[0], 'times out of', len(args), \
    '(' + str(100.0 * float(nrecommended[0]) / len(args)) + '%)'
print 'Recommended left out ingredient in top 5', nrecommended[1], 'times out of', len(args), \
    ' for random recommendations (' + str(100.0 * float(nrecommended[1]) / len(args)) + '%)'
print 'Recommended left out ingredient in top 5', nrecommended[2], 'times out of', len(args), \
    ' when recommending most common ingredients (' + str(100.0 * float(nrecommended[2]) / len(args)) + '%)'

t2 = time.clock()
print 'Took', (t2 - t1) / 60.0 ** 2, 'hours.'