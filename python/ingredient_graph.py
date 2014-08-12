__author__ = 'brandonkelly'

import numpy as np
from pmi_graph import PMIGraph, fit_pmi_graph_cv
from ingredient_mapping import IngredientMapping
import pymysql
from pymysql.err import MySQLError
import os
import cPickle


class IngredientGraph(PMIGraph):

    def __init__(self, nprior=1.0, verbose=False, database='recipes', host='localhost', user='root', passwd=''):
        super(IngredientGraph, self).__init__(nprior, verbose)
        self.database = database
        self.host = host
        self.user = user
        self.passwd = passwd
        self.imap = None
        self.ingredient_names = None

    def load_ingredient_map(self, table="Ingredient_Map"):
        self.imap = IngredientMapping().from_mysql(table, host=self.host, user=self.user, passwd=self.passwd,
                                                   database=self.database)

    def load_ingredients(self, table='Ingredient_List', limit=None):
        conn = pymysql.connect(self.host, self.user, self.passwd, self.database, charset='utf8')
        cur = conn.cursor()
        sql = "SELECT * FROM " + table
        if limit is not None:
            sql += " LIMIT " + str(limit)
        cur.execute(sql)
        rows = cur.fetchall()
        recipe_ids = []
        ingredients = []
        for id, yummly_ingredient in rows:
            recipe_ids.append(id)
            try:
                ingredients.append(self.imap[yummly_ingredient.lower()])
            except KeyError:
                ingredients.append(yummly_ingredient.lower())

        return recipe_ids, ingredients

    def build_design_matrix(self, recipe_ids, ingredients):
        unique_ingredients = np.unique(ingredients)
        self.ingredient_names = unique_ingredients
        nsamples = len(np.unique(recipe_ids))
        ningredients = len(unique_ingredients)
        X = np.zeros((nsamples, ningredients), dtype=np.int)

        ingredient_idx = dict()
        for j, ingredient in enumerate(unique_ingredients):
            ingredient_idx[ingredient] = j

        current_id = recipe_ids[0]
        r_idx = 0
        for id, ingredient in zip(recipe_ids, ingredients):
            if id != current_id:
                current_id = id
                r_idx += 1
            col_idx = ingredient_idx[ingredient]
            X[r_idx, col_idx] = 1

        return X

    def graph_to_mysql(self):
        if self.ingredient_names is None:
            raise RuntimeError("Must run self.build_design_matrix() before dumping to MySQL.")
        ningredients = len(self.ingredient_names)
        conn = pymysql.connect(self.host, self.user, self.passwd, self.database, charset='utf8', autocommit=True)
        cur = conn.cursor()
        cur.execute("DROP TABLE IF EXISTS Ingredient_Graph")
        cur.execute("CREATE TABLE Ingredient_Graph(Ingredient1 VARCHAR(200), Ingredient2 VARCHAR(200), PMI FLOAT)")
        sql = "INSERT INTO Ingredient_Graph VALUES('"
        for j in xrange(ningredients-1):
            for k in xrange(j+1, ningredients):
                sql_command = sql + self.ingredient_names[j] + "', '" + self.ingredient_names[k] + "', " + \
                    str(self.pmi[j, k]) + ")"
                try:
                    cur.execute(sql_command)
                except MySQLError:
                    print "Error when trying to insert ingredient pair (" + self.ingredient_names[j] + ', ' + \
                          self.ingredient_names[k] + ") into MySQL table. Skipping this pair."

        conn.close()


if __name__ == "__main__":
    base_dir = os.environ['HOME'] + '/Projects/Insight/'
    data_dir = base_dir + 'data/yummly/'
    plot_dir = base_dir + 'plots/'

    graph = IngredientGraph()
    graph.load_ingredient_map()
    ids, ingredients = graph.load_ingredients()
    X = graph.build_design_matrix(ids, ingredients)
    graph = fit_pmi_graph_cv(X, verbose=True, n_jobs=7, cv=7, doplot=True, graph=graph)

    X = graph.build_design_matrix(ids, ingredients)  # need to do this here since graph returned above is not the same

    cPickle.dump(graph, open(data_dir + 'ingredient_graph.pickle', 'wb'))

    graph.graph_to_mysql()