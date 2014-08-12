__author__ = 'brandonkelly'

import numpy as np
from pmi_graph import PMIGraph
from sklearn.grid_search import GridSearchCV
from ingredient_mapping import IngredientMapping
import pymysql
from pymysql.err import MySQLError
import os


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

    def load_ingredients(self, table='Ingredient_List'):
        conn = pymysql.connect(self.host, self.user, self.passwd, self.database, charset='utf8')
        cur = conn.cursor()
        cur.execute("SELECT * FROM " + table)
        rows = cur.fetchall()
        recipe_ids = []
        ingredients = []
        for id, yummly_ingredient in rows:
            recipe_ids.append(id)
            try:
                ingredients.append(self.imap[yummly_ingredient])
            except KeyError:
                ingredients.append(yummly_ingredient)

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
        for id, ingredient in zip(recipe_ids, unique_ingredients):
            if id != current_id:
                current_id = id
                r_idx += 1
            col_idx = ingredient_idx[ingredient]
            X[r_idx, col_idx] = 1

        return X

    def graph_to_mysql(self):
        conn = pymysql.connect(self.host, self.user, self.passwd, self.database)
        cur = conn.cursor()
        cur.execute("DROP TABLE IF NOT EXISTS Ingredient_Graph")
        cur.execute("CREATE TABLE Ingredient_Graph(Ingredient1 VARCHAR(200), Ingredient2 VARCHAR(200), PMI FLOAT)")
        ningredients = len(self.ingredient_names)
        sql = "INSERT INTO Ingredient_Graph VALUES('"
        for j in xrange(ningredients-1):
            for k in xrange(j+1, ningredients):
                try:
                    cur.execute(sql + self.ingredient_names[j] + "', '" + self.ingredient_names[k] + "', '" +
                                str(self.pmi[j, k]) + ")")
                except MySQLError:
                    print "Error when trying to insert ingredient pair (" + self.ingredient_names[j] + ', ' + \
                          self.ingredient_names[k] + ") into MySQL table. Skipping this pair."

        cur.close()
        conn.close()


if __name__ == "__main__":
    base_dir = os.environ['HOME'] + '/Projects/Insight/'
    data_dir = base_dir + 'data/yummly/'
    plot_dir = base_dir + 'plots/'

