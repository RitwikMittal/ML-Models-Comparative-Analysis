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

