__author__ = 'brandonkelly'

import numpy as np
from pmi_graph import PMIGraph
from sklearn.grid_search import GridSearchCV
from ingredient_mapping import IngredientMapping
import pymysql
from
import os


class IngredientGraph(PMIGraph):

    def __init__(self, nprior=1.0, verbose=False, database='recipes', host='localhost', user='root', passwd=''):
        super(IngredientGraph, self).__init__(nprior, verbose)
        self.database = database
        self.host = host
        self.user = user
        self.passwd = passwd
        self.imap = None

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
                

    def map_ingredients(self):
        pass

    def build_design_matrix(self, recipes):
        pass

    def learn_graph(self, X):
        pass

    def graph_to_mysql(self, graph):
        pass


if __name__ == "__main__":
    base_dir = os.environ['HOME'] + '/Projects/Insight/'
    data_dir = base_dir + 'data/yummly/'
    plot_dir = base_dir + 'plots/'

