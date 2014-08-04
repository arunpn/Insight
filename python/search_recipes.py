__author__ = 'brandonkelly'

import yummly
import time
import os
import cPickle
import numpy as np

base_dir = os.environ['HOME'] + '/Projects/Insight/'
data_dir = base_dir + 'data/yummly/'

api_id = '50fd16ec'
api_key = '4f425532cc37ac4ef290004ceb2e6cb3'

max_api_calls = 500
search_term = 'sauce'

included_courses0 = ['Condiments and Sauces']

excluded_courses0 = ['Lunch and Snacks', 'Salads', 'Breakfast and Brunch', 'Breads', 'Desserts', 'Beverages',
                    'Cocktails']

included_courses = []
for course in included_courses0:
    included_courses.append('course^course-' + course)

params = {'q': search_term, 'includedCourse[]': included_courses, 'maxResult': 500, 'start': 0}
print 'Search parameters:'
print params

client = yummly.Client(api_id=api_id, api_key=api_key, timeout=60.0, retries=0)

print 'Getting search results...'
matches = []
max_calls = 400
ncalls = 0
while True:  # can only return 500 at a time, otherwise yummly kicks me off
    search = client.search(**params)
    nmatches = 0
    for match in search.matches:
        if match.rating >= 4:  # only keep the good recipes
            if match.flavors['salty'] is not None:  # only keep recipes with flavor profiles
                matches.append(match)
                nmatches += 1

    print 'Found', nmatches, 'this call.'
    ncalls += 1
    print 'Did', ncalls, 'calls...'
    if ncalls == max_calls:
        break

    if len(search.matches) < 500:
        # reached the last page, no more recipes to grab
        break

    params['start'] = params['start'] + params['maxResult']  # move to next 500 recipes

print 'Found', len(matches), 'recipes.'
print 'Have', max_api_calls - ncalls, 'API calls left today.'

cPickle.dump(matches, open(data_dir + search_term + '_search.pickle', 'wb'))
