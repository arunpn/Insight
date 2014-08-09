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
IngMap.create_ingredient_map(ingredients[:10])

cPickle.dump(IngMap, open(data_dir + 'ingredient_map_backup.pickle', 'wb'))

IngMap.to_mysql("Ingredient_Map", clobber=True)