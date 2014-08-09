__author__ = 'brandonkelly'

import pymysql as mdb
import pandas as pd
import os
import numpy as np

base_dir = os.environ['HOME'] + '/Projects/Insight/'
data_dir = base_dir + 'data/yummly/'

recipes_db = mdb.connect(user="root", host="localhost", db="recipes", charset="utf8")


def bbq_sauce_piquant():
    with recipes_db:
        cur = recipes_db.cursor()
        cur.execute("SELECT * FROM Recipe_IDs")
        query_results = cur.fetchall()
    for result in query_results:
        if 'bbq' in result[1].lower():
            r_id = result[0]
            print r_id, result[1]
            break
    # grab ingredients for this recipe
    cur.execute("SELECT Ingredient FROM Ingredient_List WHERE ID = " + str(r_id))
    query_results = cur.fetchall()
    ingredients = []
    for row in query_results:
        ingredients.append(row[0])

    # find recommended ingredients
    flavors = pd.read_hdf(data_dir + 'flavor_profiles_nnls.h5', 'df')
    piquant_ingredients = flavors['type'] == 'piquant'
    graph = pd.read_hdf(data_dir + 'ingredient_graph.h5', 'df')
    # get distances between ingredients in recipe and potential additions
    graph = graph.ix[piquant_ingredients][ingredients]
    graph_distances = graph.sum(axis=1)
    recommended_ingredients = list(np.sort(graph_distances)[::-1][:5].index)

    print 'Recipe ingredients:'
    print ingredients
    print ''
    print 'Recommended Piquant ingredients:'
    print recommended_ingredients

if __name__ == "__main__":
    bbq_sauce_piquant()