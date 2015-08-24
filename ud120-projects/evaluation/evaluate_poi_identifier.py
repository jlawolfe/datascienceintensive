#!/usr/bin/python


"""
    starter code for the evaluation mini-project
    start by copying your trained/tested POI identifier from
    that you built in the validation mini-project

    the second step toward building your POI identifier!

    start by loading/formatting the data

"""

import pickle
import sys
sys.path.append("../tools/")
from feature_format import featureFormat, targetFeatureSplit

data_dict = pickle.load(open("../final_project/final_project_dataset.pkl", "r") )

### add more features to features_list!
features_list = ["poi", "salary"]

data = featureFormat(data_dict, features_list)
labels, features = targetFeatureSplit(data)



### your code goes here 
from sklearn import tree
from sklearn import cross_validation
from sklearn.metrics import precision_score, recall_score, accuracy_score


# train-test split
features_train, features_test, labels_train, labels_test = cross_validation.train_test_split(features, labels, test_size=0.3, random_state=42)

clf = tree.DecisionTreeClassifier()
clf.fit(features_train, labels_train)

# getting the accuracy when predict everything as 0.0
falseP = 0
total  = len(labels_test)
pred = clf.predict(features_test)
for i in pred:
    falseP += 1 if i > 0 else 0

print "accuracy is " + str((total - falseP) / float(total))

# check for true positive
trueP = 0
count = 0
for i in pred:
    if i == labels_test[count] and i == 1:
        trueP += 1
    count += 1

print "number of true positive is " + str(trueP)

# calculate precision_score
print "precision: " + str(precision_score(labels_test, pred)) + " vs accuracy: " + str(accuracy_score(labels_test, pred))

# calculate recall_score
print "recall: " + str(recall_score(labels_test, pred)) + " vs accuracy: " + str(accuracy_score(labels_test, pred))

# yet another test data
predictions = [0, 1, 1, 0, 0, 0, 1, 0, 1, 0, 0, 1, 0, 0, 1, 1, 0, 1, 0, 1]
true_labels = [0, 0, 0, 0, 0, 0, 1, 0, 1, 1, 0, 1, 0, 1, 1, 1, 0, 1, 0, 0]


truePositive = 0
trueNegative = 0
falsePositive = 0
falseNegative = 0
count = 0
for i in predictions:
    # get the true positive and true negatives of the data
    if i == true_labels[count]:
        if i == 1:
            truePositive += 1
        else:
            trueNegative += 1
    else:
        if i == 1: # this will be false positive
            falsePositive += 1
        else:
            falseNegative += 1
    count += 1

print "number of true positive is " + str(truePositive)
print "number of true negative is " + str(trueNegative)
print "number of false positive is " + str(falsePositive)
print "number of false negative is " + str(falseNegative)
print "precision for this test data is " + str(precision_score(true_labels, predictions))
print "recall for this test data is " + str(recall_score(true_labels, predictions))

