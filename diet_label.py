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
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.svm import LinearSVC
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.multiclass import OneVsRestClassifier
from sklearn import preprocessing



'''
edamam
{"labels": ["Vegetarian", "Peanut-Free", "Tree-Nut-Free", "Soy-Free", "Fish-Free", "Shellfish-Free", "High-Fiber", "Low-Sodium"],
	"image": "",
	"ingredients": [{"amount": 1.0, "name": "puff pastry", "unit": "sheet"},
					{"amount": 4.5, "name": "apples", "unit": "apple"},
					{"amount": 2.0, "name": "flour", "unit": "tsp"},
					{"amount": 2.0, "name": "sugar", "unit": "tsp"},
					{"amount": 4.5, "name": "apples", "unit": "medium"},
					{"amount": 1.0, "name": "lemon", "unit": "lemon"},
					{"amount": 2.0, "name": "butter", "unit": "tbsp"},
					{"amount": 1.0, "name": "egg white", "unit": "egg"}],
	 "title": "Picking apples for a free-form apple tart"}
'''

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

with open(os.getcwd()+'/data/edamam_recipes_cuisines.json') as data_file:
    data = json.load(data_file)
    data_label = []


    recipes_edaman = []

    target_names = []
    target_names_numerical = []

    for recipe_data in data:
        title = recipe_data['title']
        labels = recipe_data['labels']

        ingredients = []

        for ingredient in recipe_data['ingredients']:
            ingredients.append(ingredient['name'])

        recipe = dict()
        title_words = title.split(" ")
        recipe['data'] = ingredients + title_words
        recipe['title'] = title
        recipe['ingredients'] = ingredients
        recipe['labels'] = labels

        recipes_edaman.append(recipe)



n = int(len(recipes_edaman)*0.8)
data_train = recipes_edaman[0:n]
data_test = recipes_edaman[n:len(recipes_edaman)]


train_model_data = []

train_model_Labels = []

target = []

for recipe in data_train:

    mdata = recipe['data']
    mLabels = recipe['labels']
    train_model_data.append(' '.join(mdata))
    target.append(','.join(mLabels))
    train_model_Labels.append(mLabels)

test_model_data = []
test_model_Labels = []

for recipe in data_test:
    mdata = recipe['data']
    test_model_data.append(' '.join(mdata))

targett = []
for t in train_model_Labels:
    for tt in t:
        targett.append(tt)
#To see how many labels there are in Edaman
target_names = list(set(targett))

X_train = train_model_data
y_train_text = train_model_Labels
#X_test = ['rice risotto potatoes onion tomato garlic']

#X_test = test_model_data[0:2]

lb = preprocessing.MultiLabelBinarizer()
Y = lb.fit_transform(y_train_text)

classifier = Pipeline([
    ('vectorizer', CountVectorizer()),
    ('tfidf', TfidfTransformer()),
    ('clf', OneVsRestClassifier(LinearSVC()))])

classifier.fit(X_train, Y)

#----End models-------------
#print test_model_data[0]

with open(os.getcwd()+'/data/spoonacular_recipes_cuisines.json') as data_file:
    data = json.load(data_file)
    recipes_spoon = []

    for recipe_s in data:
        title = recipe_s['title']
        ingredients = []

        for ingredient in recipe_s['ingredients']:
            ingredients.append(ingredient['name'])

        title_words = title.split(" ")
        data_spoonacular = ingredients + title_words
        spoon_data = []
        spoon_data.append(' '.join(data_spoonacular))

        predicted = classifier.predict(spoon_data)
        all_labels = lb.inverse_transform(predicted)
        recipe_s['labels'] = all_labels

        recipes_spoon.append(recipe_s)

print "Message from diet_label"
print "Total spoonacular with DietLabel:"
print len(recipes_spoon)
print "Check if you have in your folder a new json file: spoonacular_dietLabel"

file = open(os.getcwd()+'/data/spoonacular_dietLabel.json', 'wb')
file.write(json.dumps(recipes_spoon))
file.close()

'''
print "Start---"
predicted = classifier.predict(X_test)
all_labels = lb.inverse_transform(predicted)
for item, labels in zip(X_test, all_labels):
    print '%s => %s' % (item, ', '.join(labels))
'''
