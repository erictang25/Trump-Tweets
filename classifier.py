
import csv
import re
import random
import numpy as np
from pylab import *
from numpy.matlib import repmat
import sys
import matplotlib
import matplotlib.pyplot as plt
from scipy.io import loadmat
import time
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
from datetime import datetime

# Turns time in hr:min:sec AM/PM -> time in min, ignoring secs


def time_to_min(time):
    # time, ampm = time.split()
    hr_s, min_s = time.split(':')
    hr = int(hr_s)
    min = int(min_s)
    return min + 60*hr

# Turns mo/day/yr -> days since 12/31/2014


def date_to_days(d):
    first_day = datetime(2014, 12, 31)
    mo, day, yr = d.split('/')
    mo = int(mo)
    day = int(day)
    yr = int(yr) + 2000
    date = datetime(yr, mo, day)
    delta = date-first_day
    return delta.days

# Turns mo/day/yr hr:min:sec AM/PM -> days since 12/31/2014, minutes


def date_to_days_time(dt):
    date, time = dt.split()
    return date_to_days(date), time_to_min(time)

###
# Classifier
###


# Load Train data
with open("train.csv") as csvfile:
    data = list(csv.reader(csvfile))

text_words = []
favcount = []
retweetcount = []
date = []
mins = []
for line in data[1:]:
    puncs = re.compile("[,.!?;:-]")
    line[1] = re.sub(puncs, " ", line[1])
    favcount.append(line[3])
    retweetcount.append(line[12])
    d, m = date_to_days_time(line[5])
    date.append(d)
    mins.append(m)
    words = line[1].split()
    text_words.append(words)

feature_dimension = 1000
N = len(text_words)
xTr = np.zeros((N, feature_dimension))

for x in range(0, N):
    row = np.zeros(feature_dimension)
    tweet = text_words[x]
    for word in tweet:
        row[hash(word) % feature_dimension] += 1
    xTr[x, :] = row

favcount = np.reshape(favcount, (len(favcount), 1))
retweetcount = np.reshape(retweetcount, (len(retweetcount), 1))
date = np.reshape(date, (len(retweetcount), 1))
mins = np.reshape(mins, (len(retweetcount), 1))

xTr = np.append(xTr, favcount, 1)
xTr = np.append(xTr, retweetcount, 1)
xTr = np.append(xTr, date, 1)
xTr = np.append(xTr, mins, 1)


yTr = np.zeros(N)
for i in range(0, N):
    yTr[i] = data[i+1][17]


# Load Test data
with open("test.csv") as csvfile:
    data = list(csv.reader(csvfile))

text_words = []
favcount = []
retweetcount = []
date = []
mins = []
for line in data[1:]:
    puncs = re.compile("[,.!?;:-]")
    line[1] = re.sub(puncs, " ", line[1])
    favcount.append(line[3])
    retweetcount.append(line[11])
    d, m = date_to_days_time(line[5])
    date.append(d)
    mins.append(m)
    words = line[1].split()
    text_words.append(words)

N = len(text_words)
xTe = np.zeros((N, feature_dimension))

for x in range(0, N):
    row = np.zeros(feature_dimension)
    tweet = text_words[x]
    for word in tweet:
        row[hash(word) % feature_dimension] += 1
    xTe[x, :] = row

favcount = np.reshape(favcount, (len(favcount), 1))
retweetcount = np.reshape(retweetcount, (len(retweetcount), 1))
date = np.reshape(date, (len(retweetcount), 1))
mins = np.reshape(mins, (len(retweetcount), 1))

xTe = np.append(xTe, favcount, 1)
xTe = np.append(xTe, retweetcount, 1)
xTe = np.append(xTe, date, 1)
xTe = np.append(xTe, mins, 1)

# get forest
# trees=forest(xTr,yTr,30)
clf = RandomForestClassifier(n_estimators=100, max_depth=20, random_state=0)
# a = arange(np.shape(xTr)[0])
# N = len(a)
# rand_idx = np.random.randint(len(a), size=len(a))
# xTr = xTr[rand_idx]
# yTr = yTr[rand_idx]
# # ind = np.floor(len(a)*0.8)
# # print(a)
# clf.fit(xTr[:int(N*0.8)], yTr[:int(N*0.8)])
# preds = clf.predict(xTr[int(N*0.8):])
# print("Training error: %.4f" % np.mean(preds != yTr[int(N*0.8):]))

# SVM
C_range = np.logspace(-2, 10, 13)
gamma_range = np.logspace(-9, 3, 13)
param_grid = dict(gamma=gamma_range, C=C_range)
cv = StratifiedShuffleSplit(n_splits=5, test_size=0.2, random_state=42)
grid = GridSearchCV(SVC(), param_grid=param_grid, cv=cv)
grid.fit(xTr, yTr)

print("The best parameters are %s with a score of %0.2f"
      % (grid.best_params_, grid.best_score_))

# preds = np.zeros(N)
results = ['ID,Label\n']
preds = clf.predict(xTe)
# print(preds)
for i in range(len(preds)):
    results.append(str(i) + "," + str(int(preds[i])) + "\n")
f = open("preds2.csv", "w+")
f.writelines(results)
f.close()
