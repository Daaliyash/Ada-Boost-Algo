#!/usr/bin/env python
# coding: utf-8

# # Adaboost for disease prediction

# In[62]:


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import confusion_matrix,accuracy_score,classification_report,precision_score,recall_score,f1_score
import warnings
warnings.filterwarnings('ignore')


# In[63]:


df = pd.read_csv('D:\Dowloads\Testing.csv')


# In[64]:


df


# In[65]:


df.isnull().sum()


# In[66]:


df.drop(["prognosis"],axis=1,inplace=True)
df


# In[67]:


X = df.iloc[:, :-1].values
Y = df.iloc[:, -1].values


# In[68]:


from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.45, random_state = 0)


# In[69]:


from sklearn.ensemble import AdaBoostClassifier
ada = AdaBoostClassifier(n_estimators=25, learning_rate=1.5, random_state=10)
ada.fit(X_train, Y_train)


# In[70]:


Y_pred = ada.predict(X_test)
print(np.concatenate((Y_pred.reshape(len(Y_pred),1), Y_test.reshape(len(Y_test),1)),1))


# In[71]:


print("Confusion matrix is:")
print(confusion_matrix(Y_test,Y_pred))
print("\nAccuracy score is:")
print(accuracy_score(Y_test,Y_pred))
print("\nClassification report is: ")
print(classification_report(Y_test,Y_pred))
print("\nPrecision score is:")
print(precision_score(Y_test,Y_pred))
print("\nRecall score is:")
print(recall_score(Y_test,Y_pred))
print("\nF1 score is:")
print(f1_score(Y_test,Y_pred))

