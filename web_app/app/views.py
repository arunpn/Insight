__author__ = 'brandonkelly'

from flask import render_template, jsonify, request
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

@app.route("/product", methods=['POST', 'GET'])
def index_product():
    return render_template('product.html')


def get_ingredients(yummly_url):
    pass


def get_recommendations(input_ingredients, flavor, nrecommendations):
    conn = mdb.connect('localhost', 'root', '', 'recipes', charset='utf8')
    cur = conn.cursor()
    potential_ingredients = []
    # get the list of potential ingredients to recommend
    cur.execute("SELECT DISTINCT Ingredient2 FROM Ingredient_Graph")
    rows = cur.fetchall()
    for row in rows:
        potential_ingredients.append(row[0].lower())
    ingredient_pmi = pd.Series(np.zeros(len(rows)), index=potential_ingredients)

    # find the ingredients to recommend
    if flavor != "any":
        # find subset of ingredients with this flavor profile
        sql_command = """ SELECT * FROM
                          Ingredient_Graph JOIN Ingredient_Flavors
                          WHERE Ingredient_Flavors.Type =
                          """ + flavor
        print sql_command
        cur.execute("SELECT Ingredient FROM Ingredient_Flavors WHERE Type = " + flavor)

    # compute total pairwise mutual information for the set of ingredients
    for ingredient in input_ingredients:
        sql_command = "SELECT * FROM Ingredient_Graph WHERE Ingredient1 = " + \
                      ingredient + " or Ingredient2 = " + ingredient
        cur.execute(sql_command)
        rows = cur.fetchall()
        for row in rows:
            ingredient1 = row[0].lower()
            ingredient2 = row[1].lower()
            pmi = row[3]
            if ingredient1 != ingredient:
                ingredient_pmi[ingredient1] += pmi
            else:
                ingredient_pmi[ingredient2] += pmi

    ingredient_pmi.sort(ascending=False)
    recommended_ingredients = ingredient_pmi.index[:nrecommendations]

    return recommended_ingredients


@app.route("/recommendation", methods=['GET'])
def index_recommendation():
    yummly_url = request.args.get('yummly_url')
    input_ingredients = get_ingredients(yummly_url)
    flavor_type = request.args.get('inlineRadioOptions')
    ningredients = request.args.get('ingredient_number')
    ingredients = get_recommendations(input_ingredients, flavor_type, ningredients)
    return render_template('recommendation.html', ingredients=ingredients)