#INPUT json recipes from spoonacular and edamam

import os, glob, json, requests
from pprint import pprint
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer


import logging
import numpy as np
from optparse import OptionParser
import sys
from time import time
import matplotlib.pyplot as plt

from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.linear_model import RidgeClassifier
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC
from sklearn.linear_model import SGDClassifier
from sklearn.linear_model import Perceptron
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.naive_bayes import BernoulliNB, MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neighbors import NearestCentroid
from sklearn.ensemble import RandomForestClassifier
from sklearn.utils.extmath import density
from sklearn import metrics
'''
edamam
{"calories": 2491.058395, "ingredients": [{"amount": 1.0, "name": "puff pastry", "unit": "sheet"},
                                        {"amount": 4.5, "name": "apples", "unit": "apple"},
                                        {"amount": 2.0, "name": "flour", "unit": "tsp"},
                                        {"amount": 2.0, "name": "sugar", "unit": "tsp"},
                                        {"amount": 4.5, "name": "apples", "unit": "medium"},
                                        {"amount": 1.0, "name": "lemon", "unit": "lemon"},
                                        {"amount": 2.0, "name": "butter", "unit": "tbsp"},
                                        {"amount": 1.0, "name": "egg white", "unit": "egg"}],
     "healthLabels": ["Vegetarian", "Peanut-Free", "Tree-Nut-Free", "Soy-Free", "Fish-Free", "Shellfish-Free"],
      "dietLabels": ["High-Fiber", "Low-Sodium"],
      "title": "Picking apples for a free-form apple tart"}'''


'''spoonacular
[{"ingredients": [{"category": "Produce", "amount": 1.0, "name": "bouquet garni", "unit": ""},
                {"category": "Produce", "amount": 2.0, "name": "carrots", "unit": ""},
                {"category": "Produce", "amount": 2.0, "name": "celery", "unit": "stalks"},
                {"category": "Canned and Jarred", "amount": 3.0, "name": "dried cannellini beans", "unit": "cups"},
                {"category": "Produce", "amount": 4.0, "name": "garlic", "unit": "cloves"},
                {"category": "Spices and Seasonings", "amount": 4.0, "name": "kosher salt", "unit": "servings"},
                {"category": "Canned and Jarred", "amount": 4.0, "name": "low sodium chicken stock", "unit": "cups"},
                {"category": "Oil, Vinegar, Salad Dressing", "amount": 3.0, "name": "olive oil", "unit": "tablespoons"},
                {"category": "Meat", "amount": 0.25, "name": "pancetta", "unit": "pound"},
                {"category": "Cheese", "amount": 4.0, "name": "parmigiano reggiano cheese", "unit": "servings"},
                {"category": "Produce", "amount": 1.0, "name": "red onion", "unit": ""},
                {"category": "Produce", "amount": 0.5, "name": "savoy cabbage", "unit": "head"},
                {"category": "Produce", "amount": 1.0, "name": "spinach", "unit": "cup"},
                {"category": "Produce", "amount": 0.5, "name": "swiss chard", "unit": "bunch"},
                {"category": "Canned and Jarred", "amount": 3.0, "name": "tomatoes", "unit": ""},
                {"category": "Produce", "amount": 1.0, "name": "waxy potato", "unit": ""}],
    "cuisines": ["mediterranean", "european", "italian"],
    "vegetarian": false,
    "glutenFree": true,
    "vegan": false,
    "title": "Minestrone With Parmigiano-Reggiano",
    "dairyFree": false}
'''

with open(os.getcwd()+'/data/edamam_recipes.json') as data_file:
    data = json.load(data_file)
    data_label = []


    recipes_edaman = []

    target_names = []
    target_names_numerical = []

    for recipe_data in data:
        title = recipe_data['title']

        ingredients = []

        for ingredient in recipe_data['ingredients']:
            ingredients.append(ingredient['name'])

        recipe = dict()


        title_words = title.split(" ")
        recipe['data'] = ingredients + title_words
        recipe['title'] = title
        recipe['ingredients'] = ingredients
        recipes_edaman.append(recipe)


with open(os.getcwd()+'/data/spoonacular_recipes1.json') as data_file:
    data = json.load(data_file)
    recipes = []

    target_names = []
    target_names_numerical = []

    for recipe_data in data:
        title = recipe_data['title']
        labels = recipe_data['cuisines']
        cuisines = recipe_data['cuisines']


        ingredients = []

        for ingredient in recipe_data['ingredients']:
            ingredients.append(ingredient['name'])

        recipe = dict()

        recipe['labels'] = labels
        recipe['cuisines'] = cuisines

        title_words = title.split(" ")
        recipe['data'] = ingredients + title_words
        recipe['title'] = title
        recipe['ingredients'] = ingredients

        if len(cuisines) > 0:
            target_label = cuisines[-1]
            if target_label not in target_names:
                target_names.append(target_label)

            label_index = target_names.index(target_label)
            #print label_index
            target_names_numerical.append(label_index)

            recipe['label_index'] = label_index

        recipes.append(recipe)

print "Number of recipes in spoonacular"
print len(data)


#----- Separate into label data or no
data_label = []
data_nolabel = []
for each_recipe in recipes:
    if each_recipe["cuisines"] != []:
        label = each_recipe
        label["cuisines"] = each_recipe["labels"][-1] #here is to get only one cuisne label, if I want to all cuisine level just comment this part
        l = label["cuisines"]
        l2 = l.encode('utf8')
        if l2 == "asian":
            label["cuisines"] = each_recipe['labels'][0]
        if l2 == "european":
            l = each_recipe['labels'][0]
            l3 = l.encode('utf8')
            label["cuisines"] = each_recipe['labels'][0]
            if l3 == "mediterranean":
                label["cuisines"] = each_recipe['labels'][1]
        data_label.append(label)
    else:
        nolabel = each_recipe
        data_nolabel.append(nolabel)


print "From spoonacular - Number data with cuisine"
print len(data_label)
print "From spoonacular - Number data without cuisine"
print len(data_nolabel)


n = int(len(data_label)*0.8)
data_train = data_label[0:n]
data_test = data_label[n:len(data_label)]


train_model_data = []
train_model_label = []
train_model_cuisines = []
train_model_index = []


for recipe in data_train:
    mlabel = recipe['labels']
    mdata = recipe['data']
    mcuisines = recipe['cuisines']
    mindex = recipe['label_index']
    train_model_label.append(mlabel)
    train_model_data.append(' '.join(mdata))
    train_model_cuisines.append(mcuisines)
    train_model_index.append(mindex)


test_model_data = []
test_model_label = []
test_model_cuisines = []
test_model_index = []

for recipe in data_test:
    mlabel = recipe['labels']
    mdata = recipe['data']
    mcuisines = recipe['cuisines']
    mindex = recipe['label_index']
    test_model_label.append(mlabel)
    test_model_data.append(' '.join(mdata))
    test_model_cuisines.append(mcuisines)
    test_model_index.append(mindex)

target_categorical = list(set(train_model_cuisines))
test_target_categorical = list(set(test_model_cuisines))

count_vect = CountVectorizer()
X_train_counts = count_vect.fit_transform(train_model_data)

tf_transformer = TfidfTransformer(use_idf=False).fit(X_train_counts)
X_train_tf = tf_transformer.transform(X_train_counts)
X_train_tf.shape

tfidf_transformer = TfidfTransformer()
X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)
X_train_tfidf.shape

from sklearn.naive_bayes import MultinomialNB

#clf = MultinomialNB().fit(X_train_tfidf, train_model_cuisines)
clf = LinearSVC().fit(X_train_tfidf, train_model_cuisines)

#-------------------------------------------------------
#------cuisines for edaman

demo_data = []
edaman = recipes_edaman
for recipe in edaman:
    mdata = recipe['data']
    demo_data.append(' '.join(mdata))

docs_new = demo_data
docs_new2 = ['rice risotto potatoes eggs onion tomato garlic']

X_new_counts = count_vect.transform(docs_new)
X_new_tfidf = tfidf_transformer.transform(X_new_counts)

predicted = clf.predict(X_new_tfidf)

i=0
recipes_edaman_out = []
for each_recipe in edaman:
    title = each_recipe["title"]
    ingredients = each_recipe["ingredients"]
    cuisine = predicted[i]

    recipe = dict()

    recipe['ingredients'] = ingredients
    recipe['title'] = title
    recipe['cuisines'] = cuisine
    i = i + 1

    recipes_edaman_out.append(recipe)


print "Number of recipes from edaman"
print len(recipes_edaman_out)

file = open(os.getcwd()+'/data/edamam_recipes_cuisines.json', 'wb')
file.write(json.dumps(recipes_edaman_out))
file.close()


#-----cuisines for spoonacular but the ones that the cuisnine label was empty
demo_data = []
spoonacular_nolabel = data_nolabel

for recipe in spoonacular_nolabel:
    mdata = recipe['data']
    demo_data.append(' '.join(mdata))

docs_new = demo_data
X_new_counts = count_vect.transform(docs_new)

X_new_tfidf = tfidf_transformer.transform(X_new_counts)
predicted = clf.predict(X_new_tfidf)

i=0
recipes_spoonacular_all = []

for each_recipe in spoonacular_nolabel:
    title = each_recipe["title"]
    ingredients = each_recipe["ingredients"]
    cuisine = predicted[i]

    recipe = dict()

    recipe['ingredients'] = ingredients
    recipe['title'] = title
    recipe['cuisines'] = cuisine
    i = i + 1

    recipes_spoonacular_all.append(recipe)

for each_recipe in data_label:
    title = each_recipe["title"]
    ingredients = each_recipe["ingredients"]
    cuisine = recipe['cuisines']

    recipe = dict()
    recipe['ingredients'] = ingredients
    recipe['title'] = title
    recipe['cuisines'] = cuisine
    recipes_spoonacular_all.append(recipe)

print "Total spoonacular with cuisines:"
print len(recipes_spoonacular_all)


file = open(os.getcwd()+'/data/spoonacular_recipes_cuisines.json', 'wb')
file.write(json.dumps(recipes_spoonacular_all))
file.close()

print "Check in your data folder if you have 2 json files with"
print "spoonacular_recipes_cuisines and edamam_recipes_cuisines names"
