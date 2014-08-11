__author__ = 'brandonkelly'

import numpy as np
import pymysql


class MeasurementParser(object):

    def __init__(self):
        self.query_database()

    def query_database(self):
        # initialize with MySQL query.
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

