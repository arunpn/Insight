__author__ = 'brandonkelly'

import yummly
import numpy as np
import time
import os
import cPickle
import pandas as pd

base_dir = os.environ['HOME'] + '/Projects/Insight/'
data_dir = base_dir + 'data/yummly/'

api_id = '50fd16ec'
api_key = '4f425532cc37ac4ef290004ceb2e6cb3'

client = yummly.Client(api_id=api_id, api_key=api_key, timeout=30.0, retries=0)

search = client.search('pulled pork', maxResult=500)

cPickle.dump(open(search, data_dir + 'pulled_pork_search.pickle', 'wb'))

recipes = []
print 'Fetching recipe'
for i, match in enumerate(search.matches):
    print i, '...'
    recipes.append(client.recipe(match.id))
    time.sleep(5)  # don't make too many calls to the API in rapid succession

cPickle.dump(open(recipes, data_dir + 'pulled_pork_recipes.pickle', 'wb'))
