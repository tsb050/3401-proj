import sys
import json
import random
import numpy as np

# Split a dataset into k folds
def cross_validation_split(dataset, n_folds):
	dataset_split = list()
	dataset_copy = list(dataset)
	fold_size = int(len(dataset) / n_folds)
	for i in range(n_folds):
		fold = list()
		while len(fold) < fold_size:
			index = random.randrange(len(dataset_copy))
			fold.append(dataset_copy.pop(index))
		dataset_split.append(fold)
	return dataset_split
 

# Calculate accuracy percentage
def accuracy_metric(actual, predicted):
	correct = 0
	for i in range(len(actual)):
		if actual[i] == predicted[i]:
			correct += 1
	return correct / int(len(actual)) * 100
 
def predict(tr, r, li):
    if type(tr) != list:
        return tr
    a = tr.copy()
    while type(a) == list:
        for i in range(len(li)):
            if a[0] == li[i][0]:
                break
        a = a[1][str(r[i-1])]
    return a


def accuracy(tr, m, li):
    ac = 0
    for i in range(np.size(m,axis=1)):
        if predict(tr, m[1:,i], li) == m[0,i]:
                   ac += 1
    return ac/np.size(m,axis=1)


# Evaluate an algorithm using a cross validation split
'''def evaluate_algorithm(dataset, algorithm, n_folds, *args):
	folds = cross_validation_split(dataset, n_folds)
	scores = list()
	for fold in folds:
		train_set = list(folds)
		train_set.remove(fold)
		train_set = sum(train_set, [])
		test_set = list()
		for row in fold:
			row_copy = list(row)
			test_set.append(row_copy)
			row_copy[-1] = None
		predicted = algorithm(train_set, test_set, *args)
		actual = [row[-1] for row in fold]
		accuracy = accuracy_metric(actual, predicted)
		scores.append(accuracy)
	return scores'''

#The function returns the accuracy of the decision tree when running on the test set.
def main(fname):
    testdata = np.loadtxt('../data/test.txt', dtype=int)
    with open('../data/dataDesc.txt') as f:
        e = json.load(f)
    with open('../data/'+fname) as f:
        tr = json.load(f)
    acc = accuracy(tr, testdata, e)
	return acc
