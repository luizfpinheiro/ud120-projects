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

clf = SVC(kernel='rbf', C=10000)

t_clf = time()

## Reduz o conjunto de dados para 1%
# features_train = features_train[:len(features_train)/100] 
# labels_train = labels_train[:len(labels_train)/100]

clf.fit(features_train, labels_train) ## train clf
print 'tempo de treinamento clf: ', round(time()-t_clf, 3), 's'

t_pred = time()
pred = clf.predict(features_test) ## predict values
print 'tempo de previsao: ', round(time()-t_pred, 3), 's'

# answer10 = pred[10]
# print('10 - ', answer10)
# answer26 = pred[36]
# print('26 - ', answer26)
# answer50 = pred[50]
# print('50 - ', answer50)

# Verifica quantas vezes o classificador previu que o email era de Chris
chris = 0
for test in pred:
    if test == 1:
        chris = chris + 1
print('Previsoes para Chris - ', chris)

score = clf.score(features_test, labels_test)
accuracy = accuracy_score(pred, labels_test) ## test accuracy 
print 'Precisao do classificador: ', accuracy, '%'
#########################################################


