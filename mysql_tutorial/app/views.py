__author__ = 'brandonkelly'

from flask import render_template, jsonify
from app import app
import pymysql as mdb
import pandas as pd
import os
import numpy as np

base_dir = os.environ['HOME'] + '/Projects/Insight/'
data_dir = base_dir + 'data/yummly/'

db = mdb.connect(user="root", host="localhost", db="world_innodb", charset='utf8')
recipes_db = mdb.connect(user="root", host="localhost", db="recipes", charset="utf8")

@app.route('/')
@app.route('/index')
def index():
    return render_template("index.html", title='Home', user={'nickname': 'Miguel'}, )

@app.route('/db')
def cities_page():
    with db:
        cur = db.cursor()
        cur.execute("SELECT Name FROM city LIMIT 15;")
        query_results = cur.fetchall()
    cities = ""
    for result in query_results:
        cities += result[0]
        cities += "<br>"
    return cities

@app.route("/db_fancy")
def cities_page_fancy():
    with db:
        cur = db.cursor()
        cur.execute("SELECT Name, CountryCode, Population FROM city ORDER BY Population LIMIT 15;")

        query_results = cur.fetchall()
    cities = []
    for result in query_results:
        cities.append(dict(name=result[0], country=result[1], population=result[2]))
    return render_template('cities.html', cities=cities)

@app.route("/jquery")
def index_jquery():
    return render_template('index_js.html')

@app.route("/db_json")
def cities_json():
    with db:
        cur = db.cursor()
        cur.execute("SELECT Name, CountryCode, Population FROM city")
        query_results = cur.fetchall()
    cities = []
    for result in query_results:
        cities.append(dict(City=result[0], CountryCode=result[1], Population=result[2]))

    return jsonify(dict(cities=cities))

@app.route("/mvp")
def index_mvp():
    return render_template('mvp.html')
