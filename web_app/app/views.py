from flask import Flask
app = Flask(__name__)
from flask import render_template, request
#from app import app
import pymysql as mdb
from recommendations import get_ingredients, get_recommendations

@app.route("/", methods=['POST', 'GET'])
@app.route("/index", methods=['POST', 'GET'])
def index_product():
    # home page: prompt user for a list of ingredients
    return render_template('product.html')

@app.route("/recommendation", methods=['GET'])
def index_recommendation():
    # build the ingredient recommendations page
    ingredient_input = request.args.get('ingredient_input')
    seed_recipe, unknown_ingredients = get_ingredients(ingredient_input)
    # flavor_type = request.args.get('inlineRadioOptions')
    flavor_type = 'any'
    ningredients = int(request.args.get('ingredient_number'))
    ingredients = get_recommendations(seed_recipe, flavor_type, ningredients)
    unknown_list = []
    if len(unknown_ingredients):
        # html_alert += '<div class="alert alert-warning" role="alert">'
        # html_alert += '<strong>Warning:</strong> Could not find the ingredients '
        for i, ingredient in enumerate(unknown_ingredients):
            string = ingredient
            if i == len(unknown_ingredients) - 1:
                string += '. '
            else:
                string += ', '
            unknown_list.append(unicode(string))
    print unknown_list
    if len(unknown_list) > 0:
        warning_message = True
    else:
        warning_message = False
    return render_template('recommendation.html', ingredients=ingredients, seed_recipe=seed_recipe,
                           unknown_ingredients=unknown_list, warning_message=warning_message)

@app.route("/ingredient_list")
def index_ingredients():
    # display the list of ingredients in the database when the user submits unrecognized ingredients
    recipes_db = mdb.connect(user="root", host="localhost", db="recipes", charset="utf8")
    cur = recipes_db.cursor()
    sql = "SELECT DISTINCT Ingredient2 FROM Ingredient_Graph"
    cur.execute(sql)
    rows = cur.fetchall()
    ingredients = []
    for row in rows:
        ingredients.append(row[0])

    return render_template('ingredient_list.html', ingredients=ingredients)

@app.route("/slides")
def index_slides():
    return render_template('slides.html')

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=80)
    #app.run()
