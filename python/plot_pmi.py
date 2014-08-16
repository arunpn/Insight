__author__ = 'brandonkelly'

import numpy as np
import matplotlib.pyplot as plt
import sys
import pymysql

ingredient = 'truffle oil'

conn = pymysql.connect('localhost', 'root', '', 'recipes')
cur = conn.cursor()

cur.execute("SELECT * FROM Ingredient_Graph WHERE Ingredient1 = '" +
            ingredient + "' OR Ingredient2 = '" + ingredient + "' ORDER BY PMI DESC LIMIT 15")

rows = cur.fetchall()
if len(rows) == 0:
    print "Input ingredient not found in database."
    exit()

paired_ingredient = []
pmi = []
for row in rows:
    if row[0] == ingredient:
        paired_ingredient.append(row[1])
    else:
        paired_ingredient.append(row[0])
    pmi.append(row[2])

pmi.reverse()
paired_ingredient.reverse()

yticks = 0.5 + np.arange(len(pmi))
plt.barh(yticks, pmi, align='center')
plt.xlabel("Pointwise Mutual Information")
plt.yticks(yticks, paired_ingredient)
plt.title("Most Commonly Paired Ingredients with Truffle Oil")
plt.tight_layout()
plt.show()