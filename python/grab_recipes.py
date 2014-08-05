__author__ = 'brandonkelly'

import yummly
import time
import os
import cPickle
import numpy as np
import os
import pymysql

base_dir = os.environ['HOME'] + '/Projects/Insight/'
data_dir = base_dir + 'data/yummly/'

api_id = '50fd16ec'
api_key = '4f425532cc37ac4ef290004ceb2e6cb3'

max_api_calls = 500
max_api_calls = 5
search_term = 'sauce'

client = yummly.Client(api_id=api_id, api_key=api_key, timeout=60.0, retries=0)

conn = pymysql.connect("localhost", "root", "", "recipes", autocommit=True)
cur = conn.cursor()
cur.execute("CREATE TABLE IF NOT EXISTS Yields(Id INT PRIMARY KEY, Yield VARCHAR(100))")
cur.execute("CREATE TABLE IF NOT EXISTS Recipe_Lines(Id INT, Line VARCHAR(200))")

nrows = cur.execute("SELECT * FROM Yields")

if nrows > 0:
    # find recipes that I do not have yet
    print 'Adding recipes to database...'
    current_recipes = cur.fetchall()
    cur.execute("SELECT * FROM Recipe_IDs")
    all_recipes = cur.fetchall()
    recipe_ids = []
    current_recipe_ids = []
    for id, yld in current_recipes:
        current_recipe_ids.append(id)
    for id, yid in all_recipes:
        if id not in current_recipe_ids:
            recipe_ids.append((id, yid))
else:
    # no recipes yet, so use all the IDs from the search
    print 'Creating new recipes table...'
    cur.execute("SELECT * FROM Recipe_IDs")
    recipe_ids = cur.fetchall()

print 'Fetching recipes'
pause_secs = 1
i = 0
for id, yummly_id in recipe_ids[:max_api_calls]:
    print i, '    ', id, '...', yummly_id
    recipe = client.recipe(yummly_id)
    # add this recipe to the database
    if recipe['yields'] is not None:
        # only use recipes that have a known yield, since we need this when computing the ingredient flavor profiles
        cur.execute("INSERT INTO Yields VALUES(%s, %s)", (id, recipe['yields']))
        for line in recipe['ingredientLines']:
            cur.execute("INSERT INTO Recipe_Lines VALUES(%s, %s)", (id, line))
    else:
        cur.execute("INSERT INTO Yields VALUES(%s, %s)", (id, "NULL"))
        cur.execute("INSERT INTO Recipe_Lines VALUES(%s, %s)", (id, "NULL"))

    this_pause_secs = pause_secs + np.random.uniform(-0.5, 0.5)
    time.sleep(pause_secs)  # don't make too many calls to the API in rapid succession
    i += 1