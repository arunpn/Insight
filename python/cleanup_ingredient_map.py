__author__ = 'brandonkelly'

from ingredient_mapping import IngredientMapping
import pymysql
import cPickle
import os

base_dir = os.environ['HOME'] + '/Projects/Insight/'
data_dir = base_dir + 'data/yummly/'

IngMap = cPickle.load(open(data_dir + 'ingredient_map_backup.pickle', 'rb'))

# first merge stock and broth
for ingredient in IngMap.keys():
    if 'broth' in IngMap[ingredient]:
        IngMap[ingredient] = IngMap[ingredient].replace('broth', 'stock')

# merge yoghurt/yogurt
for ingredient in IngMap.keys():
    if ('yogurt' in ingredient or 'yoghurt' in ingredient) and 'frozen' not in ingredient:
        IngMap[ingredient] = 'yogurt'

# merge mustard and dijon
for ingredient in IngMap.keys():
    if 'dijon' in IngMap[ingredient]:
        IngMap[ingredient] = 'mustard'

# merge cilantro and fresh coriander
for ingredient in IngMap.keys():
    if 'fresh coriander' in IngMap[ingredient]:
        IngMap[ingredient] = 'cilantro'

# olives -> green olives
for ingredient in IngMap.keys():
    if IngMap[ingredient] == 'olives':
        IngMap[ingredient] = 'green olives'

# pepper -> black olives
for ingredient in IngMap.keys():
    if IngMap[ingredient] == 'pepper':
        IngMap[ingredient] = 'black pepper'

# merge mustard and dijon
for ingredient in IngMap.keys():
    if 'duck sauce' in ingredient:
        IngMap[ingredient] = 'duck sauce'

# merge mozzarellas
for ingredient in IngMap.keys():
    if 'mozza' in ingredient:
        IngMap[ingredient] = 'mozzarella'

# merge the pastas
pastas = ['spaghetti', 'penne', 'bucatini', 'ravioli', 'rigatoni', 'ziti', 'rotelle', 'vermicelli', 'fusilli',
          'conchiglie', 'rotini', 'cannelloni', 'capellini', 'cavatappi', 'cavatelli', 'conchiglie', 'conchiglioni',
          'ditalini', 'fedelini', 'fettucine', 'lasagne', 'linguini', 'linguine', 'mostaccioli', 'orecchiette',
          'paccheri', 'pappardelle', 'perciatelli', 'tagliatelle', 'tortellini', 'tortiglioni', 'orzo']
for ingredient in IngMap:
    if IngMap[ingredient] in pastas:
        IngMap[ingredient] = 'pasta'

IngMap.to_mysql("Ingredient_Map", clobber=True)
cPickle.dump(IngMap, open(data_dir + 'ingredient_map.pickle', 'wb'))