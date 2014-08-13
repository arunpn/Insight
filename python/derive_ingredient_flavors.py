__author__ = 'brandonkelly'

import numpy as np
import matplotlib.pyplot as plt
import cPickle
import pandas as pd
import os
from ingredient_mapping import IngredientMapping
from scipy.optimize import nnls
from sklearn.linear_model import MultiTaskElasticNetCV, MultiTaskLassoCV
import pymysql
import argparse


def fit_nnls(X, flavors):
    flav_names = ['salty', 'sour', 'sweet', 'bitter', 'piquant', 'meaty']
    weights_mat = np.zeros((X.shape[1], flavors.shape[1]))

    for j in xrange(flavors.shape[1]):
        y = flavors[:, j]
        # check for bad values
        idx = np.isfinite(y)

        print 'Performing non-negative least-squares for ' + flav_names[j] + '...'
        weights, rnorm = nnls(X[idx], y[idx])

        weights_mat[:, j] = weights

    return weights_mat


def logit(x):
    return np.log(x / (1.0 - x))


def inv_logit(x):
    return 1.0 / (1.0 + np.exp(-x))


def fit_enet(X, flavors):
    flavors[flavors == 0] = 0.01  # logit(0) and logit(1) are not finite
    flavors[flavors == 1] = 0.99
    y = logit(flavors)
    idx = np.all(np.isfinite(y), axis=1)

    print 'Performing multi-task elastic net...'
    enet = MultiTaskElasticNetCV(cv=7, n_jobs=7, fit_intercept=False, verbose=1).fit(X, y)
    weights = inv_logit(enet.coef_.T)  # transform to 0 to 1 scale

    return weights


def fit_lasso(X, flavors):
    flavors[flavors == 0] = 0.01  # logit(0) and logit(1) are not finite
    flavors[flavors == 1] = 0.99
    y = logit(flavors)
    idx = np.all(np.isfinite(y), axis=1)

    print 'Performing multi-task elastic net...'
    lasso = MultiTaskLassoCV(cv=7, n_jobs=7, fit_intercept=False, verbose=1).fit(X, y)
    weights = inv_logit(lasso.coef_.T)  # transform to 0 to 1 scale

    return weights


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Derive ingredient profiles.")
    parser.add_argument('-m', '--model',
                        help='The model to use in the regression, either non-negative least squares (nnls), the' +
                        ' Multi-task Elastic Net (enet), or the Multi-task LASSO (lasso).',
                        default='nnls', type=str, required=False)

    args = parser.parse_args()
    model = args.model.lower()

    if model not in ['nnls', 'enet', 'lasso']:
        raise ValueError("Model must be either 'nnls', 'enet', or 'lasso'.")

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

    # construct the design matrix
    uingredients = np.unique(ingredients)
    urecipe_ids = np.unique(recipe_ids)
    X = np.zeros((len(urecipe_ids), len(uingredients)))
    print 'Found', len(urecipe_ids), 'unique recipes and', len(uingredients), 'unique ingredients.'
    for i, recipe in enumerate(urecipe_ids):
        these_ingredients = ingredients[recipe_ids == recipe]
        for ingredient in these_ingredients:
            X[i, uingredients == ingredient] = 1.0

    # hard code common ingredients, necessary to anchor the derived profiles
    salt_columns = []
    sugar_columns = []
    nuetral_columns = []
    nuetral_ingredients = ['oil', 'water', 'noodles', 'pasta', 'olive oil', 'butter', 'rice', 'rolls', 'white rice',
                           'brown rice', 'coconut oil', 'bread', 'buns', 'pasta', 'flour']
    fit_columns = []
    for j, ingredient in enumerate(uingredients):
        if ' salt' in ingredient or 'salt ' in ingredient or ' salt ' in ingredient or ingredient == 'salt':
            salt_columns.append(j)
        elif ' sugar' in ingredient or 'sugar ' in ingredient or ' sugar ' in ingredient or ingredient == 'sugar':
            sugar_columns.append(j)
        elif ' stevia' in ingredient or 'stevia ' in ingredient or ' stevia ' in ingredient or ingredient == 'stevia':
            sugar_columns.append(j)
        elif ' sweetener' in ingredient or 'sweetener ' in ingredient or ' sweetener ' in ingredient or \
                        ingredient == 'sweetener':
            sugar_columns.append(j)
        elif ' splenda' in ingredient or 'splenda ' in ingredient or ' splenda ' in ingredient or \
                        ingredient == 'splenda':
            sugar_columns.append(j)
        elif ' candy' in ingredient or 'candy ' in ingredient or ' candy ' in ingredient or \
                        ingredient == 'candy':
            sugar_columns.append(j)
        elif ingredient in nuetral_ingredients or 'olive oil' in ingredient:
            nuetral_columns.append(j)
        else:
            fit_columns.append()

    print 'Found the following salty ingredients:'
    print uingredients[salt_columns]
    print 'Found the following sugar ingredients:'
    print uingredients[sugar_columns]

    print np.sum(X.sum(axis=1) > 0)  # all recipes must have at least one ingredient
    # normalize by total number of ingredients
    X /= X.sum(axis=1)[:, np.newaxis]

    # grab the flavor profiles for each recipe
    flav_names = ['salty', 'sour', 'sweet', 'bitter', 'piquant', 'meaty']
    cur.execute("SELECT Id, Salty, Sweet, Sour, Bitter, Piquant, Meaty FROM Recipe_Attributes")
    rows = cur.fetchall()
    flavors = np.array(rows)
    flavors = flavors[urecipe_ids][:, 1:]  # first column is the recipe ID

    for salt in salt_columns:
        flavors[:, 0] -= X[:, salt]
    for sugar in sugar_columns:
        flavors[:, 1] -= X[:, sugar]
    flavors[flavors < 0] = 0.0  # keep flavor profiles positive
    remove_columns = list(salt_columns)
    remove_columns.extend(sugar_columns)
    remove_columns.extend(nuetral_columns)
    X = np.delete(X, remove_columns, axis=1)  # remove columns where we forced the salty and sweet values

    if model == 'nnls':
        weights = fit_nnls(X, flavors)
    elif model == 'enet':
        weights = fit_enet(X, flavors)
    elif model == 'lasso':
        weights = fit_lasso(X, flavors)
    weights_salt = np.zeros((len(salt_columns), flavors.shape[1]))
    weights_salt[:, 0] = 1.0
    weights_sugar = np.zeros_like(weights_salt)
    weights_sugar[:, 1] = 1.0
    weights_nuetral = np.zeros_like(weights_salt)
    weights = np.vstack((weights, weights_salt, weights_sugar, weights_nuetral))

    ingredient_names = fit_columns
    ingredient_names.extend(salt_columns)
    ingredient_names.extend(sugar_columns)
    ingredient_names.extend(nuetral_columns)

    fit, axs = plt.subplots(2, 3)
    f_idx = 0
    for row in range(2):
        for col in range(3):
            if model in ['enet', 'lasso']:
                yfit = X.dot(logit(weights[:, f_idx]))
                yfit = inv_logit(yfit)
            else:
                yfit = X.dot(weights[:, f_idx])
            axs[row, col].plot(yfit, flavors[:, f_idx], '.')
            axs[row, col].plot([0.0, 1.0], [0.0, 1.0], 'k-')
            axs[row, col].set_xlim(0, 1)
            axs[row, col].set_ylim(0, 1)
            axs[row, col].set_title(flav_names[f_idx])
            f_idx += 1
    plt.tight_layout()
    plt.show()

    counts = X > 0
    counts = counts.sum(axis=0)
    max_idx = weights.argmax(axis=1)
    itype = np.array(flav_names)[max_idx]  # is this a sweet, sour, etc. ingredient?
    data = np.hstack((counts[:, np.newaxis], weights, itype[:, np.newaxis]))

    columns = ['counts']
    columns.extend(flav_names)
    columns.append('type')

    df = pd.DataFrame(data, index=ingredient_names, columns=columns)

    df['counts'] = df['counts'].astype(np.int)
    for fname in flav_names:
        df[fname] = df[fname].astype(np.float)

    df.index.name = 'Ingredient'

    # set ingredients with all zeros to be of type 'none'
    for i in xrange(len(df)):
        if len(np.unique(df.iloc[i][flav_names])) == 1:
            # all zeros, so this is an ingredient with no flavor profile
            df.set_value(df.index[i], 'type', 'none')

    df.to_hdf(data_dir + 'flavor_profiles_' + model + '.h5', 'df')