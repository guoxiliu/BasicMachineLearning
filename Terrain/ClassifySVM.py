from class_vis import prettyPicture, output_image
from prep_terrain_data import makeTerrainData

import matplotlib.pyplot as plt
import numpy as np
import pylab as pl

features_train, labels_train, features_test, labels_test = makeTerrainData()

from sklearn.svm import SVC
clf = SVC(C=1.0, kernel="rbf")

clf.fit(features_train, labels_train)

pred = clf.predict(features_test)

from sklearn.metrics import accuracy_score
acc = accuracy_score(pred, labels_test)

print "accuracy: ", acc

prettyPicture(clf, features_test, labels_test, "test1.png")
