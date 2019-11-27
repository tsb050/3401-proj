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
        #a = a[1][str(r[i-1])]

        try:
                a = a[1][str(r[i-1])]
        except IndexError:
                a = 1
                break
        except KeyError:
                a = 1
                break

    return a


def accuracy(tr, m, li):
    ac = 0
    for i in range(np.size(m,axis=1)):
        if predict(tr, m[1:,i], li) == m[0,i]:
                   ac += 1
    return ac/np.size(m,axis=1)


#The function returns the accuracy of the decision tree when running on the test set.
def main(fname):
    testdata = np.loadtxt('../data/test.txt', dtype=int)
    with open('../data/dataDesc.txt') as f:
        e = json.load(f)
    with open('../data/'+fname) as f:
        tr = json.load(f)
    acc = accuracy(tr, testdata, e) * 100
    print('Accuracy out of 100:', acc)

        #do something

main('train_decision_tree.txt')
