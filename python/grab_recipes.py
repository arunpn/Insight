from __future__ import unicode_literals

__author__ = 'brandonkelly'

import yummly
import time
import os
import cPickle
import numpy as np
import os
import pymysql
from pymysql import err
from requests.exceptions import HTTPError

base_dir = os.environ['HOME'] + '/Projects/Insight/'
data_dir = base_dir + 'data/yummly/'

api_id = '50fd16ec'
api_key = '4f425532cc37ac4ef290004ceb2e6cb3'

max_api_calls = 5000 - 4100
search_term = 'sauce'

client = yummly.Client(api_id=api_id, api_key=api_key, timeout=120.0, retries=0)

conn = pymysql.connect("localhost", "root", "", "recipes", autocommit=True, charset='utf8')
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
    try:
        recipe = client.recipe(yummly_id)
    except HTTPError:
        continue
    # add this recipe to the database
    if recipe['yields'] is not None:
        # only use recipes that have a known yield, since we need this when computing the ingredient flavor profiles
        try:
            cur.execute("INSERT INTO Yields VALUES(%s, %s)", (id, recipe['yields']))
        except err.MySQLError:
            print "Error encountered when trying to enter Yield, just using NULL."
            cur.execute("INSERT INTO Yields VALUES(%s, %s)", (id, "NULL"))

        for line in recipe['ingredientLines']:
            try:
                cur.execute("INSERT INTO Recipe_Lines VALUES(%s, %s)", (id, line))
            except err.MySQLError:
                print "Error encountered when trying to enter recipe line, just using NULL."
                cur.execute("INSERT INTO Recipe_Lines VALUES(%s, %s)", (id, "NULL"))
    else:
        cur.execute("INSERT INTO Yields VALUES(%s, %s)", (id, "NULL"))
        cur.execute("INSERT INTO Recipe_Lines VALUES(%s, %s)", (id, "NULL"))

    this_pause_secs = pause_secs + np.random.uniform(-0.5, 0.5)
    time.sleep(pause_secs)  # don't make too many calls to the API in rapid succession
    i += 1

conn.close()