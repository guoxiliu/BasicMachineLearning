def classify(features_train, labels_train, features_test, labels_test):   
    ### import the sklearn module for GaussianNB
    ### create classifier
    ### fit the classifier on the training features and labels
    ### return the fit classifier

    from sklearn.naive_bayes import GaussianNB
    clf = GaussianNB()
    clf.fit(features_train, labels_train)
    
    pred = clf.predict(features_test)
    from sklearn.metrics import accuracy_score
    print "accuracy:", accuracy_score(pred, labels_test)
    
    return clf
    