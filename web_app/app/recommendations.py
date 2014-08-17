__author__ = 'brandonkelly'

import numpy as np
from bs4 import BeautifulSoup
from urllib import urlopen
import pymysql as mdb
from pymysql.err import MySQLError
import pandas as pd

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
        'yellow corn meal',

    ]


def get_ingredients(yummly_url):
    text_soup = BeautifulSoup(urlopen(yummly_url).read())
    tags = text_soup.findAll('meta', attrs={'property': 'yummlyfood:ingredients'})
    ingredients = []
    conn = mdb.connect('localhost', 'root', '', 'recipes')
    cur = conn.cursor()
    for tag in tags:
        yummly_ingredient = tag['content']
        # test if the ingredient is mapped to a different ingredient
        cur.execute("SELECT Ingredient FROM Ingredient_Map WHERE Yummly_Ingredient = '" + yummly_ingredient + "'")
        rows = cur.fetchall()
        if len(rows) == 0:
            # ingredient not in ingredient map, just use it's value
            ingredient = yummly_ingredient
        else:
            ingredient = rows[0][0]
        ingredients.append(ingredient)

    return ingredients


def get_recommendations(input_ingredients, flavor, nrecommendations):
    conn = mdb.connect('localhost', 'root', '', 'recipes', charset='utf8')
    cur = conn.cursor()
    potential_ingredients = []
    # get the list of potential ingredients to recommend
    cur.execute("SELECT Ingredient1 FROM Ingredient_Graph LIMIT 1")
    # need to select the first ingredient manually since it never appears as Ingredient2 in the MySQL table
    potential_ingredients.append(cur.fetchone()[0].lower())
    cur.execute("SELECT DISTINCT Ingredient2 FROM Ingredient_Graph")
    rows = cur.fetchall()
    for row in rows:
        potential_ingredients.append(row[0].lower())
    ingredient_pmi = pd.Series(np.zeros(len(potential_ingredients)), index=potential_ingredients)

    # compute total pairwise mutual information for the set of ingredients
    for ingredient in input_ingredients:
        sql_command = "SELECT * FROM Ingredient_Graph WHERE Ingredient1 = '" + \
                      ingredient + "' or Ingredient2 = '" + ingredient + "'"
        try:
            cur.execute(sql_command)
        except MySQLError:
            continue
        rows = cur.fetchall()
        if len(rows) > 0:
            # only use similarity info for ingredients used to train the ingredient graph
            for row in rows:
                ingredient1 = row[0].lower()
                ingredient2 = row[1].lower()
                pmi = row[2]
                if ingredient1 not in input_ingredients and ingredient1 not in do_not_recommend:
                    ingredient_pmi[ingredient1] += pmi
                elif ingredient2 not in input_ingredients and ingredient2 not in do_not_recommend:
                    ingredient_pmi[ingredient2] += pmi

    ingredient_pmi.sort(ascending=False)
    ingredient_pmi = ingredient_pmi[ingredient_pmi > 0]  # don't recommend statistically independent ingredients

    # find ingredients with the right flavor type
    recommended_ingredients = []
    for ingredient in ingredient_pmi.index:
        sql_command = "SELECT Type from Ingredient_Flavors WHERE Ingredient = '" + ingredient + "'"
        cur.execute(sql_command)
        row = cur.fetchall()
        if len(row) == 0:
            type = "none"
        else:
            type = row[0][0].lower()
        if type != 'none':
            if flavor != "any":
                if type == flavor:
                    recommended_ingredients.append(ingredient)
            else:
                recommended_ingredients.append(ingredient)

    recommended_ingredients = recommended_ingredients[:nrecommendations]

    return recommended_ingredients

if __name__ == "__main__":
    # test usage
    yummly_url = "http://www.yummly.com/recipe/Mole-Poblano-510445?columns=4&position=1%2F54"

    input_ingredients = get_ingredients(yummly_url)
    print "Recipe Ingredients:"
    print input_ingredients

    ftype = 'any'
    recommended_ingredients = get_recommendations(input_ingredients, ftype, 20)

    print 'Recommended ingredients for ' + ftype + ':'
    for i, ingredient in enumerate(recommended_ingredients):
        print i, '   ', ingredient

