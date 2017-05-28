from class_vis import prettyPicture
from prep_terrain_data import makeTerrainData
from sklearn import tree

import matplotlib.pyplot as plt
import numpy as np
import pylab as pl

features_train, labels_train, features_test, labels_test = makeTerrainData()


clf1 = tree.DecisionTreeClassifier(min_samples_split=2)
clf2 = tree.DecisionTreeClassifier(min_samples_split=50)
clf1.fit(features_train, labels_train)
clf2.fit(features_train, labels_train)

pred1 = clf1.predict(features_test)
pred2 = clf2.predict(features_test)
from sklearn.metrics import accuracy_score
acc_min_samples_split_2 = accuracy_score(pred1, labels_test)
acc_min_samples_split_50 = accuracy_score(pred2, labels_test)

prettyPicture(clf1, features_test, labels_test, "test1.png")
prettyPicture(clf2, features_test, labels_test, "test2.png")
print "accuracy1: ", acc_min_samples_split_2
print "accuracy2: ", acc_min_samples_split_50