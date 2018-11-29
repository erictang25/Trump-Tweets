# Eric Tang

import csv
import re
import numpy as np
from pylab import *
from numpy.matlib import repmat
import sys
import matplotlib
import matplotlib.pyplot as plt
from scipy.io import loadmat
import time

class TreeNode(object):
    """Tree class.

    (You don't need to add any methods or fields here but feel
    free to if you like. Our tests will only reference the fields
    defined in the constructor below, so be sure to set these
    correctly.)
    """

    def __init__(self, left, right, parent, cutoff_id, cutoff_val, prediction):
        self.left = left
        self.right = right
        self.parent = parent
        self.cutoff_id = cutoff_id
        self.cutoff_val = cutoff_val
        self.prediction = prediction



def sqsplit(xTr, yTr, weights=[]):
    """Finds the best feature, cut value, and loss value.

    Input:
        xTr:     n x d matrix of data points
        yTr:     n-dimensional vector of labels
        weights: n-dimensional weight vector for data points

    Output:
        feature:  index of the best cut's feature
        cut:      cut-value of the best cut
        bestloss: loss of the best cut
    """
    N, D = xTr.shape
    assert D > 0  # must have at least one dimension
    assert N > 1  # must have at least two samples
    if weights == []:  # if no weights are passed on, assign uniform weights
        weights = np.ones(N)
    weights = weights / sum(weights)  # Weights need to sum to one (we just normalize them)
    bestloss = np.inf
    feature = np.inf
    cut = np.inf

    for d in range(D):

        sorted_ind = xTr[:, d].argsort()
        sortedX = xTr[sorted_ind]
        sortedY = yTr[sorted_ind]
        sortedW = weights[sorted_ind]

        W_left = np.cumsum(sortedW)[:-1]
        W_right = np.flip(cumsum(np.flip(sortedW)))[1:]
        wy = np.multiply(sortedW, sortedY)
        P_left = np.cumsum(wy)[:-1]
        P_right = np.flip(np.cumsum(np.flip(wy)))[1:]
        P_left_sq = np.square(P_left)
        P_right_sq = np.square(P_right)
        wyy = np.multiply(sortedY, wy)
        Q_left = np.cumsum(wyy)[:-1]
        Q_right = np.flip(np.cumsum(np.flip(wyy)))[1:]

        diff_idx = np.diff(sortedX[:, d]) == 0

        loss = Q_left + Q_right - np.divide(P_left_sq, W_left) - np.divide(P_right_sq, W_right)

        loss[diff_idx] = np.inf
        min_loss_arg = np.argmin(loss)
        curr_cut = (sortedX[min_loss_arg, d] + sortedX[min_loss_arg + 1, d]) / 2
        min_loss = np.min(loss)

        if min_loss < bestloss:
            feature = d
            cut = curr_cut
            bestloss = min_loss

    return feature, cut, bestloss


def cart(xTr, yTr, depth=np.inf, weights=None):
    """Builds a CART tree.

    The maximum tree depth is defined by "maxdepth" (maxdepth=2 means one split).
    Each example can be weighted with "weights".

        self.left = left
        self.right = right
        self.parent = parent
        self.cutoff_id = cutoff_id
        self.cutoff_val = cutoff_val
        self.prediction = prediction
    Args:
        xTr:      n x d matrix of data
        yTr:      n-dimensional vector
        maxdepth: maximum tree depth
        weights:  n-dimensional weight vector for data points

    Returns:
        tree: root of decision tree
    """

    n, d = xTr.shape
    if weights is None:
        w = np.ones(n) / float(n)
    else:
        w = weights

    # TODO:
    def create_tree(parent, depth, x, y, w):
        if depth == 0 or len(y) <= 1:
            return TreeNode(None, None, parent, None, None, np.sum(np.multiply(y, w)) / np.sum(w))
        else:
            node = TreeNode(None, None, parent, 0, 0, None)
            feature, cut, _ = sqsplit(x, y, w)
            if feature == np.inf:
                # this  means all vectors are the same with diff labels
                # We set this as a leaf
                return create_tree(parent, 0, x, y, w)
            node.cutoff_id = feature
            node.cutoff_val = cut
            l_idx = (x[:, feature] < cut)
            r_idx = (x[:, feature] > cut)
            node.left = create_tree(node, depth - 1, x[l_idx], y[l_idx], w[l_idx])
            node.right = create_tree(node, depth - 1, x[r_idx], y[r_idx], w[r_idx])
            node.prediction = np.sum(np.multiply(y, w)) / np.sum(w)
            return node

    return create_tree(None, depth, xTr, yTr, w)


def evaltree(root, xTe):
    """Evaluates xTe using decision tree root.

    TreeNode():
        def __init__(self, left, right, parent, cutoff_id, cutoff_val, prediction):
        self.left = left
        self.right = right
        self.parent = parent
        self.cutoff_id = cutoff_id
        self.cutoff_val = cutoff_val
        self.prediction = prediction

    Input:
        root: TreeNode decision tree
        xTe:  n x d matrix of data points

    Output:
        pred: n-dimensional vector of predictions
    """
    assert root is not None
    n = xTe.shape[0]
    pred = np.zeros(n)

    def traverse(xTest, root):
        if root.left is None and root.right is None:
            return root.prediction
        if xTest[root.cutoff_id] > root.cutoff_val:
            return traverse(xTest, root.right)
        else:
            return traverse(xTest, root.left)

    # TODO:
    for i in range(n):
        pred[i] = traverse(xTe[i], root)

    return pred


def forest(xTr, yTr, m, maxdepth=np.inf):
    """Creates a random forest.

    Input:
        xTr:      n x d matrix of data points
        yTr:      n-dimensional vector of labels
        m:        number of trees in the forest
        maxdepth: maximum depth of tree

    Output:
        trees: list of TreeNode decision trees of length m
    """

    n, d = xTr.shape
    trees = []

    # TODO:
    for i in range(m):
        rand_idx = np.random.randint(n, size=n)
        curr_x = xTr[rand_idx]
        curr_y = yTr[rand_idx]
        trees.append(cart(curr_x, curr_y, maxdepth))

    return trees



def evalforest(trees, X, alphas=None):
    """Evaluates X using trees.

    Input:
        trees:  list of TreeNode decision trees of length m
        X:      n x d matrix of data points
        alphas: m-dimensional weight vector

    Output:
        pred: n-dimensional vector of predictions
    """
    m = len(trees)
    n, d = X.shape
    if alphas is None:
        alphas = np.ones(m) / len(trees)

    pred = np.zeros(n)

    # TODO:
    for i in range(len(trees)):
        pred += evaltree(trees[i], X) * alphas[i]

    return pred



def boosttree(x, y, maxiter=100, maxdepth=2):
    """Learns a boosted decision tree.

    Input:
        x:        n x d matrix of data points
        y:        n-dimensional vector of labels
        maxiter:  maximum number of trees
        maxdepth: maximum depth of a tree

    Output:
        forest: list of TreeNode decision trees of length m
        alphas: m-dimensional weight vector

    (note, m is at most maxiter, but may be smaller,
    as dictated by the Adaboost algorithm)
    """
    assert np.allclose(np.unique(y), np.array([-1, 1]));  # the labels must be -1 and 1
    n, d = x.shape
    weights = np.ones(n) / n
    preds = None
    forest = []
    alphas = []

    # TODO:
    for i in range(maxiter):
        new_tree = cart(x, y, maxdepth, weights)
        preds = evaltree(new_tree, x)
        preds[preds >= 0] = 1
        preds[preds < 0] = -1
        eps = np.sum(weights[np.nonzero(y - preds)]) + 1e-10
        if eps < 1 / 2:
            alpha = .5 * math.log((1 - eps) / eps)
            forest.append(new_tree)
            alphas.append(alpha)
            expon = np.exp(-1 * alpha * np.multiply(preds, y))
            weights = np.multiply(weights, expon) / (2 * math.sqrt(eps * (1 - eps)))
        else:
            return forest, alphas
    return forest, alphas

###
# Classifier
###

# Load Train data
with open("train.csv") as csvfile:
    data = list(csv.reader(csvfile))

text_words = []
for line in data[1:]:
    puncs = re.compile("[,.!?;:-]")
    line[1] = re.sub(puncs, " ", line[1])
    words = line[1].split()
    text_words.append(words)


feature_dimension = 1000
N = len(text_words)
xTr = np.zeros((N,feature_dimension))

for x in range(0,N):
    row = np.zeros(feature_dimension)
    tweet = text_words[x]
    for word in range(0,len(tweet)):
        row[hash(word) % feature_dimension] = 1
    xTr[x,:] = row

yTr = np.zeros(N)
for i in range(0,N):
    yTr[i] = data[i+1][17]


# Load Test data
with open("test.csv") as csvfile:
    data = list(csv.reader(csvfile))

text_words = []
for line in data[1:]:
    puncs = re.compile("[,.!?;:-]")
    line[1] = re.sub(puncs, " ", line[1])
    words = line[1].split()
    text_words.append(words)

N = len(text_words)
xTe = np.zeros((N, feature_dimension))

for x in range(0, N):
    row = np.zeros(feature_dimension)
    tweet = text_words[x]
    for word in range(0, len(tweet)):
        row[hash(word) % feature_dimension] = 1
    xTe[x, :] = row

# get forest
trees=forest(xTr,yTr,30)

print("Training error: %.4f" % np.mean(np.sign(evaltree(trees,xTr)) != yTr))

preds = np.zeros(N)
preds = np.sign(evalforest(trees, xTe))

predictions = np.zeros((N+1,2))
id = ["ID"]
id = id.append(np.arange(N))
label = ["Label"]
label = label.append(preds)
predictions[:,0] = id
predictions[:,1] = label
np.savetxt("preds.csv", predictions, delimiter=",")
