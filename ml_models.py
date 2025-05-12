import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import streamlit as st
from io import StringIO
from sklearn.model_selection import GridSearchCV
import joblib
from sklearn.preprocessing import StandardScaler

df = pd.read_csv("train_data.txt",header = None)
# testdf = pd.read_csv("Parkinson_Multiple_Sound_Recording/test_data.txt",header = None)
print(df.shape)

#Visulaisation
plt.figure(figsize=(25, 23))
corr = df.corr()
heatmap = sns.heatmap(corr, annot = True, square = True)

plt.savefig("Correlation Matrix Test Data.png")

xtrain = df.iloc[:,:-1]
ytrain = df.iloc[:,-1]
xtest = df.iloc[:,:-1]
ytest = df.iloc[:,-1]

## KNN
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.model_selection import cross_val_score

old_accuracy = 0
best_knn = 3

for i in range(3,10,2):
    n_neighbours= i
    knn = KNeighborsClassifier(n_neighbors = n_neighbours)
    knn.fit(xtrain, ytrain)
    ypred = knn.predict(xtest)
    ypred
    accuracy = accuracy_score(ytest,ypred)
    print("Accuracy is : ",accuracy*100,"% for KNN value = ",n_neighbours)
    if(old_accuracy < accuracy):
        old_accuracy = accuracy
        best_knn = i

print("Best KNN value = ",best_knn)
knn = KNeighborsClassifier(n_neighbors = best_knn)
knn.fit(xtrain, ytrain)
ypred = knn.predict(xtest)

target_names = ['class 0', 'class 1']
report = classification_report(ytest, ypred, target_names=target_names)
print(report)

train_accuracy = accuracy_score(ytrain, knn.predict(xtrain))
test_accuracy = accuracy_score(ytest, knn.predict(xtest))

print(f"Training Accuracy: {train_accuracy:.4f}")
print(f"Test Accuracy: {test_accuracy:.4f}")

cv_scores = cross_val_score(knn, xtrain, ytrain, cv=5)
print(f"Cross-Validation Scores: {cv_scores}")
print(f"Average CV Score: {cv_scores.mean()}")

#confusion Matrix
cm_knn = confusion_matrix(ypred,ytrain)
plt.figure(figsize=(5,5))
sns.heatmap(cm_knn,annot=True,cbar=False,cmap='crest',linewidth=2)
plt.show()

#save the model

joblib.dump(knn,'knn_parkinson.pkl')


## RANDOM FOREST

from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification
from sklearn.model_selection import cross_val_score

count = 0
rnf_accuracy = 0
best_n_estima = 100
best_maxdepth = 1
best_randomstate = 0

clf = RandomForestClassifier(criterion = "entropy")
clf.fit(xtrain,ytrain)
ypred_randomforest = clf.predict(xtest)
target_names = ['class 0', 'class 1']
report = classification_report(ytest, ypred_randomforest, target_names=target_names)
print(report)
# RandomForestClassifier(...)

param_grid = { 
    'n_estimators': range(100,300,25),
    'max_features': ['auto', 'sqrt', 'log2'],
    'max_depth' :range(1,10),
    'random_state':range(100,250,50),
    'criterion' :['gini', 'entropy']
}
CV_rfc = GridSearchCV(estimator=clf, param_grid=param_grid, cv= 5)
CV_rfc.fit(xtrain,ytrain)
CV_rfc.best_params_

rf_clf = RandomForestClassifier(criterion='entropy',
 max_depth= 15,
 max_features= 'log2',
 n_estimators= 100,
 random_state= 150)
rf_clf.fit(xtrain,ytrain)
ypred_randomforest = rf_clf.predict(xtest)
accuracy = accuracy_score(ytest,ypred_randomforest)
print("Accuracy for Random Forest =",accuracy)

target_names = ['class 0', 'class 1']
report = classification_report(ytest, ypred_randomforest, target_names=target_names)
print(report)

train_accuracy = accuracy_score(ytrain, rf_clf.predict(xtrain))
test_accuracy = accuracy_score(ytest, rf_clf.predict(xtest))

print(f"Training Accuracy: {train_accuracy:.4f}")
print(f"Test Accuracy: {test_accuracy:.4f}")

cv_scores = cross_val_score(rf_clf, xtrain, ytrain, cv=5)
print(f"Cross-Validation Scores: {cv_scores}")
print(f"Average CV Score: {cv_scores.mean()}")

#confusion Matrix
cm_knn = confusion_matrix(ypred_randomforest,ytrain)
plt.figure(figsize=(5,5))
sns.heatmap(cm_knn,annot=True,cbar=False,cmap='crest',linewidth=2)
plt.show()

#save the model

joblib.dump(rf_clf,'rf_parkinson.pkl')

## SUPPORT VECTOR MACHINE

from sklearn import svm
clf_svc = svm.SVC()
clf_svc.fit(xtrain,ytrain)

y_pred_svc = clf_svc.predict(xtest)

accuracy_svc =  accuracy_score(ytest,y_pred_svc)
print("Accuracy of SVC MODEL = ",accuracy_svc)

target_names = ['class 0', 'class 1']
report = classification_report(ytest, y_pred_svc, target_names=target_names)
print(report)

train_accuracy = accuracy_score(ytrain, clf_svc.predict(xtrain))
test_accuracy = accuracy_score(ytest, clf_svc.predict(xtest))

print(f"Training Accuracy: {train_accuracy:.4f}")
print(f"Test Accuracy: {test_accuracy:.4f}")

cv_scores = cross_val_score(clf_svc, xtrain, ytrain, cv=5)
print(f"Cross-Validation Scores: {cv_scores}")
print(f"Average CV Score: {cv_scores.mean()}")

#confusion Matrix
cm_knn = confusion_matrix(y_pred_svc,ytrain)
plt.figure(figsize=(5,5))
sns.heatmap(cm_knn,annot=True,cbar=False,cmap='crest',linewidth=2)
plt.show()

#save the model

joblib.dump(clf_svc,'SVM_parkinson.pkl')

## End