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

# pepper -> black pepper
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

# merge other misc ingredients
merge = \
    [
        ('barbecue sauce', 'bbq sauce'),
        ('barbeque sauce', 'bbq sauce'),
        ('crumbs', 'breadcrumbs'),
        ('bread crumbs', 'breadcrumbs'),
        ('beef base', 'beef bouillon'),
        ('beef stock cubes', 'beef bouillon'),
        ('chicken base', 'chiken bouillon'),
        ('chicken-flavored soup powder', 'chicken bouillon'),
        ('chile paste', 'chili paste'),
        ('chile paste with garlic', 'chili paste with garlic'),
        ('chile powder', 'chili powder'),
        ('ground chile', 'chili powder'),
        ('chili', 'chiles'),
        ('chile pepper', 'chiles'),
        ('chile sauce', 'chili sauce'),
        ('chop fine pecan', 'pecans'),
        ('catsup', 'ketchup'),
        ('pecan halves', 'pecans'),
        ('chopped bell pepper', 'bell pepper'),
        ('clove garlic, fine chop', 'garlic'),
        ('coca-cola', 'cola'),
        ('cos', 'romaine lettuce'),
        ('leav lettuc romain', 'romaine lettuce'),
        ('crabmeat', 'crab'),
        ('cream cheese, soften', 'cream cheese'),
        ('crimini', 'crimini mushrooms'),
        ('crushed pineapples in juice', 'pineapple'),
        ('curry', 'curry powder'),
        ('diced onions', 'onion'),
        ('dillweed', 'dill'),
        ('dried chile', 'dried chile peppers'),
        ('fresh green bean', 'green beens'),
        ('freshly grated parmesan', 'parmesan'),
        ('parmesan cheese', 'parmesan'),
        ('parmigiano', 'parmesan'),
        ('frozen broccoli', 'brocolli'),
        ('frozen chopped spinach, thawed and squeezed dry', 'spinach'),
        ('garbonzo', 'chickpeas'),
        ('gingerroot', 'ginger'),
        ('grated carrot', 'carrot'),
        ('green bell pepper, slice', 'green bell pepper'),
        ('green chile', 'green chilies'),
        ('ground cloves', 'clove'),
        ('whole cloves', 'clove'),
        ('dry mustard', 'mustard powder'),
        ('ground mustard', 'mustard powder'),
        ('ground sausage', 'sausage'),
        ('ground tumeric', 'tumeric'),
        ('lemon grass', 'lemongrass'),
        ('navel oranges', 'orange'),
        ('oliv pit ripe', 'green olives'),
        ('peel tomato whole', 'tomatoes'),
        ('pepper flakes', 'red pepper flakes'),
        ('evaporated milk', 'condensed milk'),
        ('powdered milk', 'condensed milk'),
        ('red bell pepper, slice', 'red bell pepper'),
        ('merlot', 'red wine'),
        ('cabernet', 'red wine'),
        ('roast red peppers, drain', 'roasted red peppers'),
        ('sausage links', 'sausages'),
        ('sesame', 'sesame seeds'),
        ('spring onions', 'green onions'),
        ('sundried tomatoes packed in olive oil', 'sun-dried tomatoes'),
        ('uncook medium shrimp, peel and devein', 'shrimp'),
        ('unsalted cashews', 'cashews'),
        ('walnut halves', 'walnuts'),
        ('walnut pieces', 'walnuts'),
        ('whipped topping', 'whipped cream'),
        ('whole kernel corn, drain', 'corn'),
        ('yellow bell pepper', 'yellow pepper')
    ]

for key, value in merge:
    IngMap[key] = value

IngMap.to_mysql("Ingredient_Map", clobber=True)
cPickle.dump(IngMap, open(data_dir + 'ingredient_map.pickle', 'wb'))