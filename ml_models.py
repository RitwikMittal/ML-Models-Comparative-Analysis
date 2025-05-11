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