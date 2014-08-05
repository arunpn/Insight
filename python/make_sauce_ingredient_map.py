__author__ = 'brandonkelly'

import pymysql
from ingredient_mapping import IngredientMapping
import cPickle

conn = pymysql.connect("localhost", "root", "", "recipes", charset='utf8')
cur = conn.cursor()
cur.execute("SELECT Ingredient FROM Ingredient_List")
rows = cur.fetchall()

ingredients = []
for row in rows:
    ingredients.append(row[0].lower())

imap = IngredientMapping()
imap.create_ingredient_map(ingredients)
print 'Found', len(imap), 'unique ingredients.'
cPickle.dump(imap, open("ingredient_map_backup.pickle", 'wb'))
imap.to_mysql("Ingredient_Map", clobber=True)