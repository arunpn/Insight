__author__ = 'brandonkelly'

import numpy as np


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
            try:
                merge.lower() in ['y', 'yes', 'n', 'no']
                break
            except ValueError:
                print('Please choose either "y" or "n".')

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
        for i in range(len(ingredients)):
            ingredients[i] = ingredients[i].lower()

        # add all of the single word ingredients to the mapping, since there should be no duplicates
        self.add_single_word_ingredients(ingredients)

        active_set = set(self.keys())
        inactive_set = set(ingredient_list) - active_set
        while len(active_set) > 0:
            # create the ingredient map by first iterating over all ingredients whose name is a single word, prompting
            # the user for duplicates, then by iterating over all ingredients whose name has two words, etc.
            for ingredient in active_set:
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
        new_ingredients = set(self.keys()) - set(ingredients)

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


if __name__ == "__main__":
    # test usage
    import cPickle
    search = cPickle.load(open('/Users/brandonkelly/Projects/Insight/data/yummly/sauce_search.pickle', 'rb'))

    ingredients = []
    for match in search:
        ingredients.extend(match.ingredients)

    sauce = ingredients.pop('sauce')
    filet = ingredients.pop('filet')
    jelly = ingredients.pop('jelly')
    juice = ingredients.pop('juice')

    IngMap = IngredientMapping()
    IngMap.create_ingredient_map(ingredients)

    cPickle.dump(IngMap, open('/Users/brandonkelly/Projects/Insight/data/yummly/ingredient_map_test.pickle', 'wb'))