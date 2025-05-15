import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix
import streamlit as st
from io import StringIO
from sklearn.model_selection import GridSearchCV
import os
os.environ['LOKY_MAX_CPU_COUNT'] = '4'
import joblib
import time


# ## DATSET LOADING AND PRE PROCESS

# In[9]:


df = pd.read_csv("Parkinson_Multiple_Sound_Recording/train_data.txt",header = None)
testdf = pd.read_csv("Parkinson_Multiple_Sound_Recording/test_data.txt",header = None)
# testdf2 = testdf.copy()


# In[10]:


testdf.head()


# In[11]:


df.describe()


# In[12]:


missing_values = df.isnull().sum()
missing_percent = (missing_values / len(df)) * 100

missing_percent


# In[13]:


df.iloc[ : ,28].value_counts()
# # df.iloc[ : ,27].nunique()


# ## Spliting Test and Train X Y

# In[14]:


from sklearn.model_selection import train_test_split

X = df.iloc[:,:-1]
y = df.iloc[:,-1]

xtrain,xtest,ytrain,ytest = train_test_split(X,y, test_size = 0.33,random_state = 45)

# xtest = testdf.iloc[:,:-1]
# ytest = testdf.iloc[:,-1]

scaler = StandardScaler()
xtrain_scale = scaler.fit_transform(xtrain)
xtest_scale = scaler.transform(xtest)

print('xtrain_scale _ shape',xtrain_scale.shape)
print('xtest_scale _ shape',xtest_scale.shape)
print('ytrain _ shape',ytrain.shape)
print('ytest _ shape',ytest.shape)


# # Visualisation

# ## Train to Test PLOTTING

# In[15]:


data = {
    'X_train': 696,
    'X_test': 344,
    'X_train_scaled': 696,
    'X_test_scaled': 344,
    'y_train': 696,
    'y_test': 344
}

labels = list(data.keys())
values = list(data.values())

# Create the bar chart
x = np.arange(len(labels))  # the label locations
width = 0.7  # the width of the bars
fig, ax = plt.subplots(figsize=(10, 6))
rects = ax.bar(x, values, width, label='Number of Instances', color=plt.cm.Blues(np.linspace(0.2, 0.8, len(values))))

# Add some text for labels, title and axes ticks
ax.set_ylabel('Number of Instances')
ax.set_xlabel('Dataset Components')
ax.set_title('Dataset Shape Overview')
ax.set_xticks(x)
ax.set_xticklabels(labels)

# Add the values on top of each bar
def autolabel(rects):
    for rect in rects:
        height = rect.get_height()
        ax.annotate('{}'.format(height),
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom')

autolabel(rects)

# Show the plot
plt.tight_layout()
plt.savefig("DatasetShapeOverview.png")
plt.show()



# ## DATA SET BOX PLOT

# In[16]:


plt.figure(figsize=(8,8))
boxplot = sns.boxplot(xtrain)
plt.savefig('boxplt.png')


# ## CORRELATION MATRIX

# In[17]:


plt.figure(figsize=(25, 23))
corr = df.corr()
heatmap = sns.heatmap(corr, annot = True, square = True)

plt.savefig("Correlation Matrix Test Data.png")


# ## PAIR PLOT

# In[18]:


pairplt = sns.pairplot(df)
plt.savefig('pairplt.png')


# ## Target Variable Histogram & Kde

# In[19]:


plt.clf()
sns.histplot(df.iloc[:,-1],kde = True,color = 'blue')
plt.ylabel("Freqency")
plt.xlabel("Value")
plt.title(f"Target Variable")
plt.savefig(f"hist n kde of Target Variable")


# ## Parameters Histogram & KDE

# In[20]:


for i in range(0,28,1):
    plt.clf()
    sns.histplot(df.iloc[:,i],kde = True,color = 'blue')
    plt.ylabel("Freqency")
    plt.xlabel("Value")
    plt.title(f"Parameter {i}")
    plt.savefig(f"hist n kde param {i}")


# In[21]:


plt.show()


# # KNN CLASSIFIER MODEL

# In[22]:


from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.model_selection import cross_val_score


# In[25]:


knn = KNeighborsClassifier()
param_grid_knn = {
    'n_neighbors': [3, 5, 7, 9],
    'weights': ['uniform', 'distance'],
    'metric': ['euclidean', 'manhattan', 'minkowski']
}


# In[26]:


clf_knn = GridSearchCV(knn, param_grid_knn, cv=5, scoring='accuracy', n_jobs=-1)

start = time.time()
clf_knn.fit(xtrain_scale, ytrain)
end = time.time()
knn_time = end - start
ypred_knn = clf_knn.predict(xtest_scale)

accuracy_knn = accuracy_score(ytest,ypred_knn)

print("Results for KNN")

target_names = ['class 0', 'class 1']
report = classification_report(ytest, ypred_knn, target_names=target_names,zero_division=0)
print(report)

train_accuracy_knn = accuracy_score(ytrain, clf_knn.predict(xtrain_scale))
test_accuracy_knn = accuracy_score(ytest, clf_knn.predict(xtest_scale))

print(f"Training Accuracy: {train_accuracy_knn:.4f}")
print(f"Test Accuracy: {test_accuracy_knn:.4f}")

cv_scores = cross_val_score(clf_knn, xtrain_scale, ytrain, cv=5)
print(f"Cross-Validation Scores: {cv_scores}")
print(f"Average CV Score: {cv_scores.mean()}")

#confusion Matrix
cm_knn = confusion_matrix(ypred_knn,ytest)
plt.figure(figsize=(5,5))
plt.title("Confusion Matrix For KNN")
sns.heatmap(cm_knn,annot=True,cbar=False,cmap='crest',linewidth=2)
plt.savefig("Knn_cfm.png")
plt.show()

#save the model

joblib.dump(clf_knn,'knn_parkinson.pkl')


# # Support Vector Machine

# In[27]:


from sklearn import svm

# param_grid_svm = {
#     'C': [0.1, 1, 10],  # Regularization parameter
#     'kernel': ['linear', 'poly', 'rbf', 'sigmoid'],
#     'gamma': ['scale', 'auto']  # Kernel coefficient
# }
# clf_svc = GridSearchCV(svc, param_grid_svm, cv=5, scoring='accuracy', n_jobs=-1)

clf_svc = svm.SVC()

start = time.time()
clf_svc.fit(xtrain_scale,ytrain)
end = time.time()
svc_time = end - start
y_pred_svc = clf_svc.predict(xtest_scale)

accuracy_svc =  accuracy_score(ytest,y_pred_svc)
print("Accuracy of SVC MODEL = ",accuracy_svc)


# In[28]:


print("Results for SVM")
target_names = ['class 0', 'class 1']
report = classification_report(ytest, y_pred_svc, target_names=target_names,zero_division=0)
print(report)

train_accuracy_svm = accuracy_score(ytrain, clf_svc.predict(xtrain_scale))
test_accuracy_svm = accuracy_score(ytest, clf_svc.predict(xtest_scale))

print(f"Training Accuracy: {train_accuracy_svm:.4f}")
print(f"Test Accuracy: {test_accuracy_svm:.4f}")

cv_scores = cross_val_score(clf_svc, xtrain_scale, ytrain, cv=5)
print(f"Cross-Validation Scores: {cv_scores}")
print(f"Average CV Score: {cv_scores.mean()}")

#confusion Matrix
cm_svc = confusion_matrix(y_pred_svc,ytest)
plt.figure(figsize=(5,5))
sns.heatmap(cm_svc,annot=True,cbar=False,cmap='crest',linewidth=2)
plt.title("Confusion Matrix For SVM")
plt.savefig("svm_cfm.png")
plt.show()

#save the model

joblib.dump(clf_svc,'SVM_parkinson.pkl')

# svc_bestparam = clf_svc.best_params_
# svc_bestacc = clf_svc.best_score_


# # LOGISTIC REGRESSION

# In[47]:


from sklearn.linear_model import LogisticRegression

clf_lr = LogisticRegression(penalty = 'l2',max_iter = 1000,random_state = 10)

start = time.time()
clf_lr.fit(xtrain_scale,ytrain)
end = time.time()
lr_time = end -start
ypred_lr = clf_lr.predict(xtest_scale)


# In[48]:


accuracy_lr = accuracy_score(ytest,ypred_lr)
print("Accuracy of Logistic Regression = ",accuracy_lr)


# In[49]:


print("Results for Logistic Regression")
target_names = ['class 0', 'class 1']
report = classification_report(ytest, ypred_lr, target_names=target_names,zero_division=0)
print(report)

train_accuracy_lr = accuracy_score(ytrain, clf_lr.predict(xtrain))
test_accuracy_lr = accuracy_score(ytest, clf_lr.predict(xtest))

print(f"Training Accuracy: {train_accuracy_lr:.4f}")
print(f"Test Accuracy: {test_accuracy_lr:.4f}")

cv_scores = cross_val_score(clf_lr, xtrain_scale, ytrain, cv=5)
print(f"Cross-Validation Scores: {cv_scores}")
print(f"Average CV Score: {cv_scores.mean()}")

#confusion Matrix
cm_knn = confusion_matrix(ypred_lr,ytest)
plt.figure(figsize=(5,5))
sns.heatmap(cm_knn,annot=True,cbar=False,cmap='crest',linewidth=2)
plt.title("Confusion Matrix For Logistic Regression")
plt.savefig("lr_cfm.png")
plt.show()

#save the model

joblib.dump(clf_svc,'LR_parkinson.pkl')


# # Navie Bayes

# In[43]:


from sklearn.naive_bayes import GaussianNB
clf_bay = GaussianNB()
start  = time.time()
clf_bay.fit(xtrain_scale,ytrain)
end = time.time()
bay_time = end - start

ypred_bay = clf_bay.predict(xtest_scale)


# In[44]:


accuracy_bay = accuracy_score(ytest,ypred_bay)
print("Accuracy of Naive Bayes = ",accuracy_bay)


# In[45]:


print("Results for Naive Bayes")
target_names = ['class 0', 'class 1']
report = classification_report(ytest, ypred_bay, target_names=target_names,zero_division=0)
print(report)

train_accuracy_bay = accuracy_score(ytrain, clf_bay.predict(xtrain))
test_accuracy_bay = accuracy_score(ytest, clf_bay.predict(xtest))

print(f"Training Accuracy: {train_accuracy_bay:.4f}")
print(f"Test Accuracy: {test_accuracy_bay:.4f}")

cv_scores = cross_val_score(clf_lr, xtrain_scale, ytrain, cv=5)
print(f"Cross-Validation Scores: {cv_scores}")
print(f"Average CV Score: {cv_scores.mean()}")

#confusion Matrix
cm_knn = confusion_matrix(ypred_bay,ytest)
plt.figure(figsize=(5,5))
sns.heatmap(cm_knn,annot=True,cbar=False,cmap='crest',linewidth=2)
plt.title("Confusion Matrix For Naive Bayes")
plt.savefig("bay_cfm.png")
plt.show()

#save the model

joblib.dump(clf_bay,'bayes_parkinson.pkl')


# In[ ]:





# # RANDOM FOREST

# In[35]:


from sklearn.ensemble import RandomForestClassifier

rf = RandomForestClassifier()
param_grid_rf = {
    'n_estimators': [100, 200, 300],  # Number of trees
    'criterion': ['gini', 'entropy'],
    'max_depth': [None,5 , 10, 20, 30] 
}
rf_clf = GridSearchCV(rf, param_grid_rf, cv=5, scoring='accuracy', n_jobs=-1)
# rf = RandomForestClassifier(criterion='entropy',
#  max_depth= 5,
#  max_features= 'log2',
#  n_estimators= 100,
#  random_state= 150)
start = time.time()
rf_clf.fit(xtrain_scale,ytrain)
end = time.time()
rf_time = end-start
ypred_randomforest = rf_clf.predict(xtest_scale)
accuracy_rf = accuracy_score(ytest,ypred_randomforest)
print("Accuracy for Random Forest =",accuracy_rf)


# In[36]:


print("Results for Random Forest")
target_names = ['class 0', 'class 1']
report = classification_report(ytest, ypred_randomforest,zero_division=0, target_names = target_names)
print(report)

train_accuracy_rf = accuracy_score(ytrain, rf_clf.predict(xtrain_scale))
test_accuracy_rf = accuracy_score(ytest, rf_clf.predict(xtest_scale))

print(f"Training Accuracy: {train_accuracy_rf:.4f}")
print(f"Test Accuracy: {test_accuracy_rf:.4f}")

cv_scores = cross_val_score(rf_clf, xtrain_scale, ytrain, cv=5)
print(f"Cross-Validation Scores: {cv_scores}")
print(f"Average CV Score: {cv_scores.mean()}")

#confusion Matrix
cm_knn = confusion_matrix(ypred_randomforest,ytest, labels = [0,1])
plt.figure(figsize=(5,5))
sns.heatmap(cm_knn,annot=True,cbar=False,cmap='crest',linewidth=2)
plt.title("Confusion Matrix For Random Forest")
plt.savefig("rf_cfm.png")
plt.show()

#save the model

joblib.dump(rf_clf,'rf_parkinson.pkl')


# In[ ]:





# # Final Comaparitive Results & Evaluation

# In[37]:


print("The Classification Models used are: \n1. K Nearest Neighbours\n2. Random Forest\n3. Support Vector Machine\n4. Logistic Regression\n5. Naive Bayes")
print("\nAccuracy Comparison : \n")
models = ["K Nearest Neighbours", 'Random Forest' , 'Support Vector Machine' , 'Logistic Regression', 'Naive Bayes']
accuracy = [accuracy_knn, accuracy_rf, accuracy_svc, accuracy_lr, accuracy_bay]
index = 0
for name in models:
    print(f"Accuracy of {name} is : {(accuracy[index])*100}%")
    index += 1


# In[38]:


print("\nTraining Time Comparison : \n")
models = ["K Nearest Neighbours", 'Random Forest' , 'Support Vector Machine' , 'Logistic Regression', 'Naive Bayes']
accuracy = [knn_time, rf_time, svc_time, lr_time, bay_time]
index = 0
for name in models:
    print(f"Computational Time of {name} is : {(accuracy[index])*100}s")
    index += 1


# ## Training vs Validation Accuracy Plot

# In[39]:


models = ['KNN', 'SVM' , 'Logistic Regression','Naive Bayes', 'Random Forest']
training_accuracy = [train_accuracy_knn,train_accuracy_svm,train_accuracy_lr,train_accuracy_bay,train_accuracy_rf]
validation_accuracy = [test_accuracy_knn,test_accuracy_svm,test_accuracy_lr,test_accuracy_bay,test_accuracy_rf]

# Plotting
bar_width = 0.2
index = np.arange(len(models))

plt.figure(figsize=(10,6))
plt.bar(index, training_accuracy, bar_width, label='Training Accuracy', color='blue')
plt.bar(index + bar_width, validation_accuracy, bar_width, label='Validation Accuracy', color='red')

# Customization
plt.xlabel('Models')
plt.ylabel('Accuracy')
plt.title('Training vs Validation Accuracy Across Models')
plt.xticks(index + bar_width / 2, models, rotation=15)
plt.ylim(0, 1.05)
plt.legend()
plt.tight_layout()
plt.show()
plt.savefig('Train vs valid accuracy.png')


# ## ROC AUC Plot for Model Metrics

# ## CV SCORES PLOT

# In[58]:


import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc

# Define model names and corresponding classifiers
model_names = ["KNN", "SVM", "Logistic Regression", "Naïve Bayes", "Random Forest"]
models = [clf_knn, clf_svc, clf_lr, clf_bay, rf_clf]

plt.figure(figsize=(10, 6))

for model_name, model in zip(model_names, models):
    if model_name == "SVM":
        y_scores = model.decision_function(xtest_scale)  # Use decision function for SVM
    else:
        y_scores = model.predict_proba(xtest_scale)[:, 1]  # Use probability estimates for other models

    fpr, tpr, _ = roc_curve(ytest, y_scores)  # Compute ROC curve
    roc_auc = auc(fpr, tpr)  # Compute AUC score

    plt.plot(fpr, tpr, label=f"{model_name} (AUC = {roc_auc:.2f})")

# Plot baseline reference line (random classifier)
plt.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Random Classifier")

# Customize the plot
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC-AUC Curve Comparison Across Models")
plt.legend()
plt.grid()

# Show the plot
plt.show()
plt.savefig('ROC.png')
# Loop through models to generate ROC-AUC curves
# for model_name, model in zip(model_names, models):
#     y_probs = model.predict_proba(xtest_scale)[:, 1]  # Get probability estimates
#     fpr, tpr, _ = roc_curve(ytest, y_probs)  # Compute ROC curve
#     roc_auc = auc(fpr, tpr)  # Compute AUC score
    
#     plt.plot(fpr, tpr, label=f"{model_name} (AUC = {roc_auc:.2f})")


# In[51]:


from sklearn.metrics import classification_report
import pandas as pd

# Define model names and corresponding classifiers
model_names = ["KNN", "SVM", "Logistic Regression", "Naïve Bayes", "Random Forest"]
models = [clf_knn, clf_svc, clf_lr, clf_bay, rf_clf]

# Storage for classification reports
classification_reports = {}

# Generate classification reports for each model
for name, model in zip(model_names, models):
    y_pred = model.predict(xtest_scale)  # Make predictions
    report = classification_report(ytest, y_pred, output_dict=True, zero_division=0)  # Get structured report
    classification_reports[name] = report

# Convert reports into a structured DataFrame
df_metrics = pd.DataFrame.from_dict({name: classification_reports[name]["weighted avg"] for name in model_names}, orient="index")

# Display the classification report table
print(df_metrics)

# Plot bar chart
plt.figure(figsize=(12, 6))
x = np.arange(len(model_names))
width = 0.2  # Bar width for grouping

# plt.bar(x - width*1.5, df_metrics["Aaccuracy"], width, label="Accuracy", color="blue")
plt.bar(x - width/2, df_metrics["precision"], width, label="Precision", color="green")
plt.bar(x + width/2, df_metrics["recall"], width, label="Recall", color="orange")
plt.bar(x + width*1.5, df_metrics["f1-score"], width, label="F1-score", color="red")

# Customize the chart
plt.xticks(x, model_names, rotation=45)
plt.ylabel("Score")
plt.title("Comparison of Model Performance Metrics")
plt.legend()
plt.grid(axis="y", linestyle="--")

# Show the plot
plt.show()


# ## Training Time Comparison

# In[52]:


model_names = ["KNN", "SVM", "Logistic Regression", "Naïve Bayes", "Random Forest"]
accuracy = [knn_time, svc_time, lr_time, bay_time,rf_time]

model_dict = dict(zip(model_names, accuracy))

labels = list(model_dict.keys())
values = list(model_dict.values())

# Create the bar chart
x = np.arange(len(labels))  # the label locations
width = 0.7  # the width of the bars
fig, ax = plt.subplots(figsize=(10, 6))
rects = ax.bar(x, values, width, label='Number of Instances', color=plt.cm.Blues(np.linspace(0.2, 0.8, len(values))))

# Add some text for labels, title and axes ticks
ax.set_ylabel('Number of Instances')
ax.set_xlabel('Dataset Components')
ax.set_title('Model Training Time Comparison')
ax.set_xticks(x)
ax.set_xticklabels(labels)


def autolabel(rects):
    for rect in rects:
        height = rect.get_height()
        ax.annotate('{}'.format(height),
                    xy=(rect.get_x() + rect.get_width() / 2, rect.get_y()),  # Position at the base
                    xytext=(0, -12),  # Offset downwards
                    textcoords="offset points",
                    ha='center', va='bottom')  # Align text at the top of annotation
ax.set_xticklabels(labels, rotation=15, ha='right')

autolabel(rects)
plt.ylim(0.0003,10)
# Show the plot
plt.tight_layout()
plt.savefig("DatasetShapeOverview.png")
plt.show()