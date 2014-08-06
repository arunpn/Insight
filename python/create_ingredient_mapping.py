__author__ = 'brandonkelly'

from ingredient_mapping import IngredientMapping
import pymysql

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

IngMap.to_mysql("Ingredient_Map", clobber=True)