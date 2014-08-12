__author__ = 'brandonkelly'

import numpy as np
import pymysql


class MeasurementParser(object):
    # valid units of measurement
    units = ['cup', 'cups', 'teaspoon', 'tablespoon', 'tablespoons', 'teaspoons', 'ounces', 'tsp.', 'tsp', 'pound',
             'tbsp', 'Tbs.', 'lb', 'tbsp.', 'pounds', 'Tbsp', 'lb.', 'Tbsp.', 'cup(s)', 'Tablespoons', 'pinch', 'oz',
             'Tablespoon', 'tablespoon(s)', 'T', 'bunch', 'g', 'lbs', 'dash', 'oz.', 'ounce', 'ml', 'gr.',
             'sprigs', 'Cup', 'pint', 'Tbs', 'TBS', 'dashes', 'grams', 'quarts', 'tbsps.', 'lbs.', 'drops', 'wedges',
             'C', 'c', 'pound(s)', 'Tb.', 'c.', 'ounce(s)', 't', 'tbs', 'Teaspoon']

    # map to unique measurement units
    units_map = {'cup': 'cup', 'cups': 'cup', 'teaspoon': 'tsp', 'teaspoons': 'tsp', 'tablespoon': 'tbsp',
                 'tablespoons': 'tbsp', 'ounces': 'oz', 'tsp.': 'tsp', 'pound': 'lb', 'tbsp': 'tbsp', 'Tbs.': 'tbsp',
                 'lb': 'lb', 'tbsp.': 'tbsp', 'pounds': 'lb', 'Tbsp': 'tbsp', 'lb.': 'lb', 'Tbsp.': 'tbsp',
                 'cup(s)': 'cup', 'Tablespoons': 'tbsp', 'pinch': 'pinch', 'oz': 'oz', 'Tablespoon': 'tbsp',
                 'tablespoon(s)': 'tbsp', 'T': 'tbsp', 'bunch': 'bunch', 'g': 'g', 'lbs': 'lb',
                 'dash': 'dash', 'oz.': 'oz', 'ounce': 'oz', 'ml': 'ml', 'gr.': 'g', 'sprigs': 'sprig', 'Cup': 'cup',
                 'pint': 'pint', 'Tbs': 'tbsp', 'TBS': 'tbsp', 'dashes': 'dashes', 'grams': 'g', 'quarts': 'quart',
                 'lbs.': 'lbs', 'drops': 'drops', 'wedges': 'wedges', 'C': 'cup', 'c': 'cup',
                 'pound(s)': 'lb', 'Tb.': 'tbsp', 'c.': 'cup', 'ounce(s)': 'oz', 't': 'tsp', 'tbs': 'tbsp',
                 'Teaspoon': 'tsp'}

    # convert measurement units to standard volume units
    unit_conversion = {}

    def __init__(self):
        # initialize with MySQL query.
        self.ids = []
        self.lines = []
        self.query_database()

    def query_database(self):
        conn = pymysql.connect('localhost', 'root', '', 'recipes')
        cur = conn.cursor()
        cur.execute("SELECT * FROM Recipe_Lines")
        rows = cur.fetchall()
        ids = []
        lines = []
        for id, line in rows:
            ids.append(id)
            lines.append(line)
        self.ids = ids
        self.lines = lines
        cur.close()
        conn.close()

    def find_numerical_measurements(self):
        """
        Find the recipes that have a numerical value for each ingredient amount.

        :param rows: The rows returned by the MySQL query.
        :return: The list of recipe IDs with clean ingredient amounts.
        """
        ids = []
        last_id = self.ids[0]
        bad_lines = 0
        for id, line in zip(self.ids, self.lines):
            first = line.split()[0]
            if id == last_id:
                if not first[0].isdigit():
                    # assume that if the first character is a number than we have a good measurement
                    bad_lines += 1
            else:
                if bad_lines == 0:
                    ids.append(last_id)
                last_id = id
                bad_lines = 0

        return ids

    def get_volumes(self):
        pass