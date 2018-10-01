#!/usr/bin/python

""" 
    This is the code to accompany the Lesson 2 (SVM) mini-project.

    Use a SVM to identify emails from the Enron corpus by their authors:    
    Sara has label 0
    Chris has label 1
"""
    
import sys
from time import time
sys.path.append("../tools/")
from email_preprocess import preprocess


### features_train and features_test are the features for the training
### and testing datasets, respectively
### labels_train and labels_test are the corresponding item labels
features_train, features_test, labels_train, labels_test = preprocess()




#########################################################
### your code goes here ###
from sklearn.svm import SVC 
from sklearn.metrics import accuracy_score

clf = SVC(kernel='linear')

t_clf = time()
clf.fit(features_train, labels_train) ## train clf
print 'tempo de treinamento clf: ', round(time()-t_clf, 3), 's'

t_pred = time()
pred = clf.predict(features_test) ## predict values
print 'tempo de previsao: ', round(time()-t_pred, 3), 's'

score = clf.score(features_test, labels_test)
accuracy = accuracy_score(pred, labels_test) ## test accuracy 
print 'Precisao do classificador: ', accuracy, '%'
#########################################################


