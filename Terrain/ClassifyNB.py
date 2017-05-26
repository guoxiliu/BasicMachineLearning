def classify(features_train, labels_train):   
    ### import the sklearn module for GaussianNB
    ### create classifier
    ### fit the classifier on the training features and labels
    ### return the fit classifier

    from sklearn.naive_bayes import GaussianNB
    clf = GaussianNB()
    clf.fit(features_train, labels_train)
    return clf
    

def NBAccuracy(classifier, features_test, labels_test):
    pred = classifier.predict(features_test)
    from sklearn.metrics import accuracy_score
    print accuracy_score(pred, labels_test)