
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
from sklearn.model_selection import KFold
from datetime import datetime


# Turns time in hr:min:sec AM/PM -> time in min, ignoring secs
from sklearn.naive_bayes import GaussianNB, MultinomialNB
from sklearn.svm import SVC
from textblob import TextBlob
import nltk
# nltk.download('brown')
# nltk.download('punkt')
# nltk.download('averaged_perceptron_tagger')


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
polarity = []
subjectivity = []
crookedHillary = []
numnouns = []
pos = []
fakenews = []
for line in data[1:]:
    puncs = re.compile("[,?!.;:]")
    line[1] = re.sub(puncs, " ", line[1])
    # line[1] = line[1].lower()

    if 'Crooked Hillary' in line[1]:
        crookedHillary.append(1)
    else:
        crookedHillary.append(0)
    if 'fake news' in line[1]:
        fakenews.append(1)
    else:
        fakenews.append(0)

    favcount.append(line[3])
    text = TextBlob(line[1])
    numnouns.append(len(text.noun_phrases))
    pos.append(text.pos_tags)
    polarity.append(text.sentiment.polarity)
    subjectivity.append(text.sentiment.subjectivity)
    retweetcount.append(line[12])
    d, m = date_to_days_time(line[5])
    date.append(d)
    mins.append(m)
    words = line[1].split()
    text_words.append(words)


feature_dimension = 1000
N = len(text_words)
xTr = np.zeros((N,feature_dimension))

for x in range(0,N):
    row = np.zeros(feature_dimension)
    tweet = text_words[x]
    tweetLen = len(tweet)
    pos_row = np.zeros(50)
    for word in tweet:
        row[hash(word) % feature_dimension] = 1

    for word,tag in pos[x]:
        pos_row[hash(tag) % 50] = 1

    xTr[x,:] = row
    np.append(xTr[x,:], pos_row)
    np.append(xTr[x, :], tweetLen)

favcount = np.reshape(favcount, (len(favcount),1))
retweetcount = np.reshape(retweetcount, (len(retweetcount),1))
date = np.reshape(date, (len(retweetcount),1))
mins = np.reshape(mins, (len(retweetcount),1))
polarity = np.reshape(polarity, (len(retweetcount),1))
subjectivity = np.reshape(subjectivity, (len(retweetcount),1))
numnouns = np.reshape(numnouns, (len(retweetcount), 1))
crookedHillary = np.reshape(crookedHillary, (len(retweetcount), 1))
fakenews = np.reshape(fakenews, (len(retweetcount), 1))

xTr = np.append(xTr, favcount, 1)
xTr = np.append(xTr, retweetcount, 1)
xTr = np.append(xTr, date, 1)
xTr = np.append(xTr, mins, 1)
xTr = np.append(xTr, polarity, 1)
xTr = np.append(xTr, subjectivity, 1)
xTr = np.append(xTr, numnouns, 1)
xTr = np.append(xTr, crookedHillary, 1)
xTr = np.append(xTr, fakenews, 1)

yTr = np.zeros(N)
for i in range(0,N):
    yTr[i] = data[i+1][17]


# Load Test data
with open("test.csv") as csvfile:
    data = list(csv.reader(csvfile))

text_words = []
favcount = []
retweetcount = []
date = []
mins = []
polarity = []
subjectivity = []
numnouns = []
pos = []
crookedHillary = []
fakenews = []
for line in data[1:]:
    puncs = re.compile("[,.?!;:]")
    line[1] = re.sub(puncs, " ", line[1])
    # line[1] = line[1].lower()

    if 'Crooked Hillary' in line[1]:
        crookedHillary.append(1)
    else:
        crookedHillary.append(0)
    if 'fake news' in line[1]:
        fakenews.append(1)
    else:
        fakenews.append(0)
    favcount.append(line[3])
    text = TextBlob(line[1])
    numnouns.append(len(text.noun_phrases))
    pos.append(text.pos_tags)
    polarity.append(text.sentiment.polarity)
    subjectivity.append(text.sentiment.subjectivity)
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
    pos_row = np.zeros(50)
    tweetLen = len(tweet)
    for word in tweet:
        row[hash(word) % feature_dimension] = 1
    for word,tag in pos[x]:
        pos_row[hash(tag) % 50] += 1
    xTe[x, :] = row
    np.append(xTe[x, :], pos_row)
    np.append(xTe[x, :], tweetLen)

favcount = np.reshape(favcount, (len(favcount),1))
retweetcount = np.reshape(retweetcount, (len(retweetcount),1))
date = np.reshape(date, (len(retweetcount),1))
mins = np.reshape(mins, (len(retweetcount),1))
polarity = np.reshape(polarity, (len(retweetcount),1))
subjectivity = np.reshape(subjectivity, (len(retweetcount),1))
numnouns = np.reshape(numnouns, (len(retweetcount), 1))
crookedHillary = np.reshape(crookedHillary, (len(retweetcount), 1))
fakenews = np.reshape(fakenews, (len(retweetcount), 1))

xTe = np.append(xTe, favcount, 1)
xTe = np.append(xTe, retweetcount, 1)
xTe = np.append(xTe, date, 1)
xTe = np.append(xTe, mins, 1)
xTe = np.append(xTe, polarity, 1)
xTe = np.append(xTe, subjectivity, 1)
xTe = np.append(xTe, numnouns, 1)
xTe = np.append(xTe, crookedHillary, 1)
xTe = np.append(xTe, fakenews, 1)


# get forest
a = arange(np.shape(xTr)[0])
N = len(a)

clf = RandomForestClassifier(n_estimators=1000, max_depth=None, random_state=0, max_features="sqrt")

rand_idx = np.random.randint(len(a), size=len(a))
xTr = xTr[rand_idx]
yTr = yTr[rand_idx]

kf = KFold(n_splits=4,shuffle=True, random_state=None)
for train_index, test_index in kf.split(xTr):
    clf.fit(xTr[train_index], yTr[train_index])
    preds = clf.predict(xTr[test_index])
    print("Validation error: %.4f" % np.mean(preds != yTr[test_index]))

clf.fit(xTr, yTr)
preds = clf.predict(xTr)
print("Validation error: %.4f" % np.mean(preds != yTr))

results = ['ID,Label\n']
preds = clf.predict(xTe)
for i in range(len(preds)):
    results.append(str(i) + "," + str(int(preds[i])) + "\n")
f = open("preds13.csv", "w+")
f.writelines(results)
f.close()

