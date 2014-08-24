__author__ = 'brandonkelly'

import numpy as np
from pmi_graph import PMIGraph, fit_pmi_graph_cv
from ingredient_mapping import IngredientMapping
import pymysql
from pymysql.err import MySQLError
import os
import cPickle
import matplotlib.pyplot as plt

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


class IngredientGraph(PMIGraph):

    def __init__(self, nprior=1.0, verbose=False, database='recipes', host='localhost', user='root', passwd=''):
        """
        Constructor for the Ingredient graph class. The ingredient graph provides a representation of the association
        among ingredients, where the nodes represent the ingredients and the edges provide a measure of the chances
        that two ingredients appear in the same recipe. Specifically, the edges represent the pointwise mutual
        information between two ingredients.

        :param nprior: The prior sample size, i.e., the graph shrinkage parameter. See the documentation on PMIGraph for
            further details.
        :param verbose: Print helpful information?
        :param database: The name of the database containing the recipes.
        :param host: The host of the database.
        :param user: The name of a user that can access the database.
        :param passwd: The password of the user.
        """
        super(IngredientGraph, self).__init__(nprior, verbose)
        self.database = database
        self.host = host
        self.user = user
        self.passwd = passwd
        self.imap = None
        self.ingredient_names = None

    def load_ingredient_map(self, table="Ingredient_Map"):
        """
        Load the object that maps ingredients to a set of base ingredients and save internally as a data member.

        :param table: The name of the MySQL table containing the ingredient mapping.
        """
        self.imap = IngredientMapping().from_mysql(table, host=self.host, user=self.user, passwd=self.passwd,
                                                   database=self.database)

    def load_ingredients(self, table='Ingredient_List', limit=None):
        """
        Load the ingredients used in each recipe, along with the corresponding recipe IDs, from the MySQL database.

        :param table: The name of the table containing the list of ingredients and their recipe IDs.
        :param limit: If limit is not none, then only this many rows will be returned from the MySQL database.
        :return: A tuple containing the list of recipes IDs and ingredients.
        """
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

    def build_design_matrix(self, recipe_ids, ingredients, min_counts=2):
        """
        Construct the design matrix used in learning the graph from the input set of recipes and ingredients.

        :param recipe_ids: A list containing the IDs of the recipes for each ingredients.
        :param ingredients: A list containing the ingredients for each recipe.
        :param min_counts: The minimum number of recipes that an ingredient must appear in before including it in the
            graph.
        :return: The design matrix, and (nrecipes, ningredients) array where nrecipes is the number of unique recipes
            and ningredients is the number of unique ingredients appearining in at least min_counts recipes.
        """
        if len(recipe_ids) != len(ingredients):
            raise ValueError("Length of recipe_ids and ingredients must by the same.")

        unique_ingredients = np.unique(ingredients)
        frequent_ingredients = []
        for j, ingredient in enumerate(unique_ingredients):
            if ingredients.count(ingredient) > min_counts:
                # only consider ingredient that have appeared in some minimum number of recipes
                frequent_ingredients.append(ingredient)

        self.ingredient_names = frequent_ingredients
        nsamples = len(np.unique(recipe_ids))
        ningredients = len(frequent_ingredients)
        X = np.zeros((nsamples, ningredients), dtype=bool)

        ingredient_idx = dict()
        for j, ingredient in enumerate(frequent_ingredients):
            ingredient_idx[ingredient] = j

        current_id = recipe_ids[0]
        r_idx = 0
        for id, ingredient in zip(recipe_ids, ingredients):
            if id != current_id:
                current_id = id
                r_idx += 1
            if ingredient in frequent_ingredients:
                col_idx = ingredient_idx[ingredient]
                X[r_idx, col_idx] = True

        return X

    def graph_to_mysql(self):
        """
        Dump the graph to a MySQL database.
        """
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

    def cluster(self, normalize=False):
        """
        Cluster the ingredients based on their pointwise mutual information as a similarity measure. See the
        documentation of PMIGraph for further info.

        :param normalize: If true, then normalize the PMI to be between -1 and 1.
        :return: The cluster labels.
        """
        clusters = super(IngredientGraph, self).cluster(normalize)
        if self.verbose:
            cluster_ids = np.unique(clusters)
            for c_id in cluster_ids:
                cluster_info = 'Cluster ' + str(c_id) + ':'
                this_cluster = np.array(self.ingredient_names)[clusters == c_id]
                for ingredient in this_cluster:
                    cluster_info += ' ' + ingredient + ','
                print cluster_info

        return clusters

    def visualize(self, cluster=False, savefile=None, doshow=True, seed=None, node_labels=None, label_idx=None,
                  mark_nodes=False, make_graph_illustration=False, nnodes=5):
        """
        Visualize the graph structure. The similarity matrix is converted into a 2-d representation of the nodes using
        the t-distributed stochastic neighbor embedding algorithm.

        :param cluster: If true, then also cluster the ingredients.
        :param savefile: The name of a file for saving the plot to.
        :param doshow: The true, then display the plot in a window.
        :param seed: The random number generator seed.
        :param node_labels: The labels of a subset of nodes to display.
        :param label_idx: The indices of the labeled nodes.
        :param mark_nodes: If true, mark the nodes supplied by node_labels and label_idx.
        :param make_graph_illustration: If true, illustrate how the graph works by marking the nodes for an example
            input recipe. This is only used to make the slides illustrating what the backend of theflavory.me does.
        :param nnodes: The number of nodes to label if not explicitly provided.
        :return: The matplotlib axis instance and node positions.
        """
        if node_labels is None or label_idx is None:
            random_idx = np.random.permutation(len(self.ingredient_names))
            label_idx = []
            node_labels = []
            for idx in random_idx:
                if self.ingredient_names[idx] not in do_not_recommend:
                    node_labels.append(self.ingredient_names[idx])
                    label_idx.append(idx)
                    if len(node_labels) == nnodes:
                        break
        ax, node_positions = \
            super(IngredientGraph, self).visualize(cluster, savefile, doshow, seed, node_labels, label_idx, mark_nodes)

        if make_graph_illustration:
            node_labels = ['chicken', 'garlic', 'cream', 'pasta']
            node_idx = []
            for i, label in enumerate(node_labels):
                node_idx.append(np.where(np.array(graph.ingredient_names) == label)[0])
                plt.scatter(node_positions[0, node_idx[i]], node_positions[1, node_idx[i]], s=500, c='Green')
                plt.text(node_positions[0, node_idx[i]] + 0.02 * node_positions[0].ptp(),
                         node_positions[1, node_idx[i]] + 0.02 * node_positions[1].ptp(),
                         label, size=20, color='White')

            recom_labels = ['gorgonzola', 'vodka', 'prosciutto', 'parmigiano', 'marsala']
            recom_idx = []
            for i, label in enumerate(recom_labels):
                recom_idx.append(np.where(np.array(graph.ingredient_names) == label)[0])
                for node in node_idx:
                    plt.plot([node_positions[0, recom_idx[i]], node_positions[0, node]],
                             [node_positions[1, recom_idx[i]], node_positions[1, node]], '-', color='Magenta', lw=3)
                plt.scatter(node_positions[0, node_idx], node_positions[1, node_idx], s=500, c='Green')
                plt.scatter(node_positions[0, recom_idx[i]], node_positions[1, recom_idx[i]], s=500, c='Orange')
                plt.text(node_positions[0, recom_idx[i]] + 0.02 * node_positions[0].ptp(),
                         node_positions[1, recom_idx[i]] + 0.02 * node_positions[1].ptp(),
                         label, size=20, color='White')

            plt.savefig(plot_dir + 'recommendation_graph.png', facecolor='k', edgecolor='Yellow')
            plt.show()

        return ax, node_positions


if __name__ == "__main__":
    base_dir = os.environ['HOME'] + '/Projects/Insight/'
    data_dir = base_dir + 'data/yummly/'
    plot_dir = base_dir + 'plots/'

    load_pickle = True
    if load_pickle:
        graph = IngredientGraph()
        graph = cPickle.load(open(data_dir + 'ingredient_graph.pickle', 'rb'))
        node_labels = ['chicken', 'garlic', 'cream', 'pasta']
        node_idx = []
        for label in node_labels:
            node_idx.append(np.where(np.array(graph.ingredient_names) == label)[0])
        graph.visualize(cluster=False, seed=213, node_labels=[], label_idx=[],
                        savefile=plot_dir + 'ingredient_graph.png', doshow=False, make_graph_illustration=True)
        exit()
    doCV = False
    if doCV:
        graph = IngredientGraph()
        print 'Loading the ingredient map...'
        graph.load_ingredient_map()
        print 'Loading recipes...'
        ids1, ingredients1 = graph.load_ingredients(table='Ingredient_List_Graph')
        print 'Loading recipes for sauces and condiments...'
        ids2, ingredients2 = graph.load_ingredients()  # ingredient labels as sauces or condiments
        ids2 = list(np.array(ids2) + np.max(ids1) + 1)
        ids1.extend(ids2)
        ingredients1.extend(ingredients2)
        X = graph.build_design_matrix(ids1, ingredients1, min_counts=50)
        print 'Found', X.shape[1], 'ingredients and', X.shape[0], 'recipes.'
        graph = fit_pmi_graph_cv(X, verbose=True, n_jobs=7, cv=7, doplot=True, graph=graph)
        # need to do this here since graph returned above is not the same
        X = graph.build_design_matrix(ids1, ingredients1, min_counts=50)
    else:
        graph = IngredientGraph(verbose=True, nprior=1e4)
        graph.load_ingredient_map()
        ids1, ingredients1 = graph.load_ingredients(table='Ingredient_List_Graph')
        ids2, ingredients2 = graph.load_ingredients()  # ingredient labels as sauces or condiments
        ids2 = list(np.array(ids2) + np.max(ids1) + 1)
        ids1.extend(ids2)
        ingredients1.extend(ingredients2)
        X = graph.build_design_matrix(ids1, ingredients1, min_counts=50)
        print 'Found', X.shape[1], 'ingredients and', X.shape[0], 'recipes.'
        graph.fit(X)
    cPickle.dump(graph, open(data_dir + 'ingredient_graph.pickle', 'wb'))
    print 'Saving Graph to MySQL...'
    graph.graph_to_mysql()