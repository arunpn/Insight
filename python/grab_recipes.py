__author__ = 'brandonkelly'

import yummly
import time
import os
import cPickle
import numpy as np
import os

base_dir = os.environ['HOME'] + '/Projects/Insight/'
data_dir = base_dir + 'data/yummly/'

api_id = '50fd16ec'
api_key = '4f425532cc37ac4ef290004ceb2e6cb3'

max_api_calls = 500 - 164
max_api_calls = 50
search_term = 'sauce'

client = yummly.Client(api_id=api_id, api_key=api_key, timeout=60.0, retries=0)

matches = cPickle.load(open(data_dir + search_term + '_search.pickle', 'rb'))

if os.path.isfile(data_dir + search_term + '_recipes.pickle'):
    # find recipes that I do not have yet
    recipes = cPickle.load(open(data_dir + search_term + '_recipes.pickle', 'rb'))
    recipe_ids = []
    current_recipe_ids = []
    for recipe in recipes:
        current_recipe_ids.append(recipe.id)
    for match in matches:
        if match.id not in current_recipe_ids:
            recipe_ids.append(match.id)
else:
    # no recipes yet, so use all the IDs from the search
    recipe_ids = []
    for match in matches:
        recipe_ids.append(match.id)
    recipes = []

print 'Fetching recipes'
pause_secs = 1
for i, id in enumerate(recipe_ids[:max_api_calls]):
    print i, '    ', id, '...'
    recipes.append(client.recipe(id))
    this_pause_secs = pause_secs + np.random.uniform(-0.5, 0.5)
    time.sleep(pause_secs)  # don't make too many calls to the API in rapid succession

cPickle.dump(recipes, open(data_dir + search_term + '_recipes.pickle', 'wb'))
