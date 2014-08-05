from __future__ import unicode_literals
__author__ = 'brandonkelly'

import numpy as np
import pymysql


class IngredientMapping(dict):

    ignore_ingredients = ['sauce', 'filet', 'jelly', 'juice']

    def merge_ingredient_pair(self, ingredient, base_ingredient):
        """
        Prompt the user for input on whether two ingredients are really the same thing. If they are, then the ingredient
        map will be updated to reflect this.

        :param ingredient: The ingredient in question.
        :param base_ingredient: The base ingredient, for which ingredient is suspected of being equivalent to.
        """
        merge = ''
        while True:
            merge = raw_input('Are ' + ingredient + ' and ' + base_ingredient + ' the same [y/n]? ')
            if not merge.lower() in ['y', 'yes', 'n', 'no']:
                print('Please choose either "y" or "n".')
                continue
            else:
                break

        if merge.lower() in ['y', 'yes']:
            # the two ingredients are the same
            self[ingredient] = base_ingredient
        else:
            self[ingredient] = ingredient

        return

    def add_single_word_ingredients(self, ingredient_list):
        """
        Add all ingredients to the mapping whose name is a single word.

        :param ingredient_list: The list of ingredients. All ingredients whose name is a single word will be added to
            the mapping.
        """
        for ingredient in ingredient_list:
            words = ingredient.split()
            if len(words) == 1:
                self[ingredient] = ingredient

    def consolidate_ingredients(self, ingredients, base_ingredient):
        """
        Consolidate an input list of ingredients by comparing against the input base ingredient. This method checks
        for duplicate ingredients that have different names.

        :param ingredients: The list of ingredients to compare with base_ingredient for duplicates.
        :param base_ingredient: The ingredient against which the list will be compared.
        """
        base_words = base_ingredient.split()
        for ingredient in ingredients:
            ingredient_words = ingredient.split()
            word_count = 0
            for word in base_words:
                # check if all words in the base ingredient name are also in the longer ingredient name. If true, we
                # have a potential duplication
                word_count += word in ingredient_words
            if word_count == len(base_words) and base_ingredient != ingredient:
                # found a potential match: the name of ingredient in contained within the name of base_ingredient
                if ingredient in self.keys():
                    # make sure this ingredient is not already mapped to a different ingredient
                    if ingredient == self[ingredient]:
                        if base_ingredient not in self.ignore_ingredients:
                            self.merge_ingredient_pair(ingredient, base_ingredient)
                else:
                    if base_ingredient not in self.ignore_ingredients:
                        self.merge_ingredient_pair(ingredient, base_ingredient)

        return

    def create_ingredient_map(self, ingredient_list):
        """
        Create the mapping between ingredient name and the ingredient.

        :param ingredient_list: The list of ingredients to base the mapping on.
        """
        ingredients = np.unique(ingredient_list)
        print 'Found', len(ingredients), 'unique ingredients.'
        for i in range(len(ingredients)):
            ingredients[i] = ingredients[i].lower()

        # add all of the single word ingredients to the mapping, since there should be no duplicates
        self.add_single_word_ingredients(ingredients)

        active_set = set(self.keys())
        inactive_set = set(ingredient_list) - active_set
        while len(active_set) > 0:
            # create the ingredient map by first iterating over all ingredients whose name is a single word, prompting
            # the user for duplicates, then by iterating over all ingredients whose name has two words, etc.
            for i, ingredient in enumerate(active_set):
                print 'Doing active set ingredient', i, 'out of', len(active_set), '...'
                self.consolidate_ingredients(inactive_set, ingredient)
            nwords = len(ingredient.split())
            # find all ingredients with names containing one more word than the current active set. this is the new
            # active set
            active_set = set()
            for ingredient in inactive_set:
                if len(ingredient.split()) == nwords + 1:
                    active_set.add(ingredient)

        # find all stragglers and add them to the ingredient list
        added_ingredients = set(self.keys())
        missing_ingredients = set(ingredient_list) - added_ingredients
        for ingredient in missing_ingredients:
            self[ingredient] = ingredient

        return

    def add_ingredients(self, ingredient_list):

        ingredients = np.unique(ingredient_list)
        for i in range(len(ingredients)):
            ingredients[i] = ingredients[i].lower()

        # find which of the new ingredients have not been seen before
        new_ingredients = set(ingredients) - set(self.keys())
        print 'Found', len(ingredients), 'unique ingredients, of which', len(new_ingredients), 'are new.'

        # add the new ingredients
        for ingredient in new_ingredients:
            self[ingredient] = ingredient

        # find all of the ingredients that map to the same name, test for duplicates
        potential_matches = set()
        for ingredient in self.keys():
            if self[ingredient] == ingredient:
                potential_matches.add(ingredient)

        # add to the ingredient map by first iterating over all ingredients whose name is a single word, prompting
        # the user for duplicates, then by iterating over all ingredients whose name has two words, etc.

        active_set = set()
        for ingredient in potential_matches:
            nwords = len(ingredient.split())
            if nwords == 1:
                active_set.add(ingredient)

        inactive_set = potential_matches - active_set
        while len(active_set) > 0:

            for ingredient in active_set:
                self.consolidate_ingredients(inactive_set, ingredient)
            nwords = len(ingredient.split())
            # find all ingredients with names containing one more word than the current active set. this is the new
            # active set
            active_set = set()
            for ingredient in inactive_set:
                if len(ingredient.split()) == nwords + 1:
                    active_set.add(ingredient)

        return

    def map_ingredients(self, ingredients):
        """
        Map the input ingredient list onto the ingredient map.

        :param ingredients: A list of ingredient names.
        :return: A list of the ingredient names, corrected for duplicates.
        """
        mapped = []
        for ingredient in ingredients:
            if ingredient in self.keys():
                mapped.append(self[ingredient])
            else:
                mapped.append(ingredient)

        return mapped

    def to_mysql(self, table, clobber=False, host='localhost', user='root', passwd='', database='recipes'):
        """
        Store the ingredient mapping to a MySQL database. If the database already exists, then any values not already
         in the database will be added.

        :param table: The name of the table containing the ingredient map.
        :param clobber: If true, then delete the current table, if it exists.
        """
        conn = pymysql.connect(host, user, passwd, database, autocommit=True, charset='utf8')
        cur = conn.cursor()
        if clobber:
            cur.execute("DROP TABLE IF EXISTS " + table)
            cur.execute("CREATE TABLE " + table +
                        "(Yummly_Ingredient VARCHAR(100) PRIMARY KEY, Ingredient VARCHAR(100))")
        else:
            # check if table exists. if it does not, create a new table
            cur.execute("CREATE TABLE IF NOT EXISTS " + table +
                        "(Yummly_Ingredient VARCHAR(100) PRIMARY KEY, Ingredient VARCHAR(100))")

        cur.execute("SELECT Yummly_Ingredient FROM " + table)
        rows = cur.fetchall()
        yingredients = []
        for row in rows:
            yingredients.append(row[0])
        new_ingredients = set(self.keys()) - set(yingredients)
        print 'Found', len(new_ingredients), 'ingredients not in the MySQL database. Adding them...'
        for ingredient in new_ingredients:
            cur.execute("INSERT INTO " + table + " VALUES('" + ingredient + "', '" + self[ingredient] + "')")


if __name__ == "__main__":
    # test usage
    conn = pymysql.connect('localhost', 'root', '', 'recipes', charset='utf8')
    cur = conn.cursor()
    cur.execute("SELECT Ingredient from Ingredient_List LIMIT 1000")
    rows = cur.fetchall()
    ingredients = []
    for row in rows:
        ingredients.append(row[0].lower())

    ingredients1 = ingredients[0:50]
    ingredients2 = ingredients[50:100]

    IngMap = IngredientMapping()
    print 'Creating ingredient map for first half...'
    IngMap.create_ingredient_map(ingredients1)
    print 'Adding ingredients from second half...'
    IngMap.add_ingredients(ingredients2)

    IngMap0 = IngredientMapping()
    print 'Creating ingredient mapping for all ingredients...'
    IngMap0.create_ingredient_map(ingredients[:100])

    ingredients1 = set(IngMap.keys())
    ingredients2 = set(IngMap0.keys())
    assert len(ingredients2 - ingredients1) == 0
    assert len(ingredients1 - ingredients2) == 0

    for ingredient in ingredients1:
        assert IngMap[ingredient] == IngMap0[ingredient]

    IngMap.to_mysql("Ingredient_Map", clobber=False)