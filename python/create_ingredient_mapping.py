__author__ = 'brandonkelly'

from ingredient_mapping import IngredientMapping
import pymysql
import cPickle
import os

base_dir = os.environ['HOME'] + '/Projects/Insight/'
data_dir = base_dir + 'data/yummly/'

conn = pymysql.connect('localhost', 'root', '', 'recipes', charset='utf8')
cur = conn.cursor()
cur.execute("SELECT Ingredient from Ingredient_List")
rows = cur.fetchall()
ingredients = []
for row in rows:
    ingredients.append(row[0].lower())

IngMap = IngredientMapping()
print 'Creating ingredient mapping for all ingredients...'
IngMap.create_ingredient_map(ingredients)

cPickle.dump(IngMap, open(data_dir + 'ingredient_map_backup.pickle', 'wb'))

IngMap.to_mysql("Ingredient_Map", clobber=True)

# NOTES:
# merge stock and broth
# merge the pastas
# merge the yoghurts/yogurt
# merge mustard and dijon
# merge olives and green olives
# rename pepper to black pepper
# incorrectly mapped duck sauce -> duck, fix all duck sauces
# merge fresh coriander and cilantro
# merge mozzarellas

# ingredients to ignore when recommending recipes:
ignore_ingredients = ['buns', 'lamb', 'halibut', 'flour', 'swordfish', 'bread', 'chuck', 'fish', 'salmon',
                      'yellowtail', 'bass', 'pudding', 'tilapia', 'pork', 'doughnuts', 'lettuce', 'pastry',
                      'chicken', 'steak', 'water', 'cod', 'turkey', 'beef', 'sirloin', 'ravioli', 'noodles',
                      'rib', 'tuna', 'veal', 'vinaigrette', 'margarine', 'flavoring', 'candy', 'trout', 'wheat',
                      'tofu', 'tenderloin', 'loin', 'rolls', 'vegetables', 'cake', 'meatballs', 'baguette', 'ham',
                      'sausages', 'pizza crust', 'white rice', 'brown rice', 'pasta']