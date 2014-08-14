from __future__ import unicode_literals

__author__ = 'brandonkelly'

import yummly
import time
import os
import cPickle
import numpy as np
from requests.exceptions import HTTPError
import pymysql

base_dir = os.environ['HOME'] + '/Projects/Insight/'
data_dir = base_dir + 'data/yummly/'

api_id = '50fd16ec'
api_key = '4f425532cc37ac4ef290004ceb2e6cb3'

max_api_calls = 5000 - 3
search_term = 'sauce'

excluded_courses0 = ['Breads', 'Beverages', 'Cocktails', 'Condiments and Sauces']

excluded_courses = []
for course in excluded_courses0:
    excluded_courses.append('course^course-' + course)

params = {'q': search_term, 'excludedCourse[]': excluded_courses, 'maxResult': 500, 'start': 0}

client = yummly.Client(api_id=api_id, api_key=api_key, timeout=120.0, retries=0)

# connect to the mysql server and create the tables
conn = pymysql.connect('localhost', 'root', '', 'recipes', autocommit=True, charset='utf8')
cur = conn.cursor()
cur.execute("DROP TABLE IF EXISTS Recipe_IDs_Graph")
cur.execute("CREATE TABLE Recipe_IDs_Graph(Id INT NOT NULL, YummlyID VARCHAR(1000), PRIMARY KEY(Id))")
cur.execute("DROP TABLE IF EXISTS Ingredient_List_Graph")
cur.execute("CREATE TABLE Ingredient_List_Graph(Id INT, Ingredient VARCHAR(200))")

print 'Getting search results...'
matches = []
max_matches = 100000
nmatches_total = 0
ncalls = 0

debug = False
if debug:
    # don't waste API calls while debugging
    matches = cPickle.load(open(data_dir + search_term + '_search.pickle', 'rb'))
    matches = matches[:10]
    match_id = 0
    for match in matches:
        print match.id
        print match_id
        cur.execute("INSERT INTO Recipe_IDs_Graph(Id, YummlyID) VALUES(%s, %s)", (match_id, match.id))
        for ingredient in match.ingredients:
            cur.execute("INSERT INTO Ingredient_List_Graph VALUES(%s, %s)", (match_id, ingredient))
        match_id += 1

    exit()

match_id = 0
nsaved = 0
while nmatches_total < max_matches:  # can only return 500 at a time, otherwise yummly kicks me off
    try:
        search = client.search(**params)
    except HTTPError:
        continue

    nmatches_total += len(search.matches)
    print 'Search returned', len(search.matches), 'matches this call.'
    print nmatches_total, 'total matches returned thus far.'
    nmatches = 0
    for match in search.matches:
        if match.flavors['salty'] is not None:  # only keep recipes with flavor profiles
            cur.execute("INSERT INTO Recipe_IDs_Graph(Id, YummlyID) VALUES(%s, %s)", (match_id, match.id))
            for ingredient in match.ingredients:
                cur.execute("INSERT INTO Ingredient_List_Graph VALUES(%s, %s)", (match_id, ingredient))
            match_id += 1
            nmatches += 1
            nsaved += 1

    print 'Found', nmatches, 'this call. New total is', nsaved
    ncalls += 1
    print 'Did', ncalls, 'calls...'

    params['start'] += len(search.matches)

print 'Found', len(matches), 'recipes.'
print 'Have', max_api_calls - ncalls, 'API calls left today.'

conn.close()
