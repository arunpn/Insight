__author__ = 'brandonkelly'

from flask import render_template
from app import app
import pymysql as mdb

db = mdb.connect(user="root", host="localhost", db="world_innodb", charset='utf8')

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
