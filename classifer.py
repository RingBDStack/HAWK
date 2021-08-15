# -*- coding: utf-8 -*-
# import pandas as pd
import matplotlib
import numpy as np
import sklearn
matplotlib.rcParams['font.sans-serif'] = [u'simHei']
matplotlib.rcParams['axes.unicode_minus'] = False
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
# from xgboost import XGBClassifier

print("Sklearn verion is {}".format(sklearn.__version__))


M = np.load('get_new_emb\\app_lab_emb.npz.npy')
y = M[:, 0]


X = np.load('getembeding\\featts.npy')

# X_test = np.load('F:\\python\\out-of-sample\\new_app_emb_CIC2019\\new_app_emb_all_6.npz.npy')
# y_test1 = np.ones((len(X_test)-100,))
# y_test2 = np.zeros((100,))
# y_test = np.concatenate((y_test1,y_test2))
# print(y_test.shape)
#########################################################################
# lmin=3448
# lmax=3547
# emin=7086
#
# emax=7535
#
# M=np.load('F:\\python\\metapath2vec\label_emb_M\\app_emb_label2015.npz.npy')
# X= np.load('F:\\python\\out-of-sample\\base_out_sample\\han\\featts2015.npy')
# y=M[:, 0]
#
# X_train1=X[0:lmin]
# X_train2=X[lmax+1:emin]
# X_train3=X[emax+1:]
# X_train = np.concatenate((X_train1,X_train2))
# X_train = np.concatenate((X_train,X_train3))
#
# y_train1=y[0:lmin]
# y_train2=y[lmax+1:emin]
# y_train3=y[emax+1:]
# y_train = np.concatenate((y_train1,y_train2))
# y_train = np.concatenate((y_train,y_train3))
#
#
# X_test1=X[lmin:lmax+1]
# X_test2=X[emin:emax+1]
# X_test = np.concatenate((X_test1,X_test2))
#
# y_test1=y[lmin:lmax+1]
# y_test2=y[emin:emax+1]
# y_test = np.concatenate((y_test1,y_test2))

##############################################################
# M=np.load('F:\\python\\metapath2vec\label_emb_M\\app_emb_label2019.npz.npy')
# X= np.load('F:\\python\\out-of-sample\\base_out_sample\\han\\featts2019.npy')
# y=M[:, 0]
#
#
#
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.95)


print("==========================================")
RF = RandomForestClassifier(n_estimators=10, random_state=11)
RF.fit(X_train, y_train)
predictions = RF.predict(X_test)
print("RF")
print(classification_report(y_test, predictions,digits=4))
print("AC", accuracy_score(y_test, predictions))


### Logistic Regression Classifier
# print("==========================================")
# from sklearn.linear_model import LogisticRegression
#
# clf = LogisticRegression(penalty='l2')
# clf.fit(X_train, y_train)
# predictions = clf.predict(X_test)
# print("LR")
# print(classification_report(y_test, predictions,digits=4))
# print("AC", accuracy_score(y_test, predictions))

### Decision Tree Classifier
print("==========================================")
from sklearn import tree

clf = tree.DecisionTreeClassifier()
clf.fit(X_train, y_train)
predictions = clf.predict(X_test)
print("DT")
print(classification_report(y_test, predictions,digits=4))
print("AC", accuracy_score(y_test, predictions))

# ### GBDT(Gradient Boosting Decision Tree) Classifier
# print("==========================================")
# from sklearn.ensemble import GradientBoostingClassifier
#
# clf = GradientBoostingClassifier(n_estimators=200)
# clf.fit(X_train, y_train)
# predictions = clf.predict(X_test)
# print("GBDT")
# print(classification_report(y_test, predictions,digits=4))
# print("AC", accuracy_score(y_test, predictions))
#
# ###AdaBoost Classifier
# print("==========================================")
# from sklearn.ensemble import AdaBoostClassifier
#
# clf = AdaBoostClassifier()
# clf.fit(X_train, y_train)
# predictions = clf.predict(X_test)
# print("AdaBoost")
# print(classification_report(y_test, predictions,digits=4))
# print("AC", accuracy_score(y_test, predictions))
#
# ### GaussianNB
# print("==========================================")
# from sklearn.naive_bayes import GaussianNB
#
# clf = GaussianNB()
# clf.fit(X_train, y_train)
# predictions = clf.predict(X_test)
# print("GaussianNB")
# print(classification_report(y_test, predictions,digits=4))
# print("AC", accuracy_score(y_test, predictions))

## Linear Discriminant Analysis
print("==========================================")
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

clf = LinearDiscriminantAnalysis()
clf.fit(X_train, y_train)
predictions = clf.predict(X_test)
print("Linear Discriminant Analysis")
print(classification_report(y_test, predictions,digits=4))
print("AC", accuracy_score(y_test, predictions))

# ### Quadratic Discriminant Analysis
# print("==========================================")
# from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
#
# clf = QuadraticDiscriminantAnalysis()
# clf.fit(X_train, y_train)
# predictions = clf.predict(X_test)
# print("Quadratic Discriminant Analysis")
# print(classification_report(y_test, predictions))
# print("AC", accuracy_score(y_test, predictions))

### SVM Classifier
print("==========================================")
from sklearn.svm import SVC

clf = SVC(kernel='rbf', probability=True)
clf.fit(X_train, y_train)
predictions = clf.predict(X_test)
print("SVM")
print(classification_report(y_test, predictions,digits=4))
print("AC", accuracy_score(y_test, predictions))



