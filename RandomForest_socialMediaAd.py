#!/usr/bin/env python
# coding: utf-8

# In[15]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.tree import DecisionTreeClassifier

import warnings
warnings.filterwarnings('ignore')


# In[3]:


df = pd.read_csv('Social_Network_Ads.csv')
df.head()


# In[4]:


df.info()


# In[5]:


df.isna().sum()


# In[12]:


x = df.iloc[:,0:2]
y = df.iloc[:,2]


# In[13]:


# Splitting the dataset into train and test
from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=42)


# ##  Hyperparameter Tunning

# In[17]:


from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier


# In[18]:


rf = RandomForestClassifier(random_state=42,n_jobs=1)


# In[19]:


params = {
    'max_depth': [2,6,5,10,22],
    'min_samples_leaf': [3,5,10,15,20,50,100],
    'n_estimators': [10,20,30,50,100,200],
    'min_samples_split': [8, 10, 12,16]

}


# In[20]:


# Instantiate the grid search model
grid_search = GridSearchCV(estimator=rf,
                           param_grid=params,
                           cv = 4,
                           n_jobs=-1, verbose=1, scoring="accuracy")


# In[21]:


grid_search.fit(x_train, y_train)


# In[22]:


grid_search.best_score_


# In[23]:


rf_best = grid_search.best_estimator_
rf_best


# In[24]:


rf_best.feature_importances_


# In[25]:


imp_df = pd.DataFrame({
    "Varname": x_train.columns,
    "Imp": rf_best.feature_importances_
})


# In[26]:


imp_df.sort_values(by="Imp", ascending=False)


# # Random Forest Classifier using K-Fold

# In[30]:


from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score

num_trees = 100
kfold = KFold(n_splits=8, random_state=42, shuffle=True) #Bootstrap

model = RandomForestClassifier(n_estimators=20,max_depth=2,min_samples_leaf=3)
results = cross_val_score(model, x, y, cv=kfold)
print(results.mean())


# # Random Forest Classifier using Train-Test split

# In[31]:


from sklearn.ensemble import RandomForestClassifier
model2 = RandomForestClassifier(max_depth=2, min_samples_leaf=3, min_samples_split=8,
                       n_estimators=20, n_jobs=1, random_state=42) 
model2 = model2.fit(x_train, y_train)


# In[33]:


y_train_pred = model2.predict(x_train)
y_test_pred = model2.predict(x_test)


# In[34]:


#confusion matrix of train data
from sklearn.metrics import confusion_matrix,classification_report
confusion_matrix(y_train,y_train_pred)
sns.heatmap(confusion_matrix(y_train,y_train_pred),annot=True, fmt='d',cbar=False, cmap='rainbow')
plt.xlabel('Predicted Label')
plt.ylabel('Actual Label')
plt.title('Confusion Matrix')
plt.show()


# In[35]:


#classification report of train data
print(classification_report(y_train,y_train_pred))


# In[36]:


#confusion matrix of test data
confusion_matrix(y_test,y_test_pred)
sns.heatmap(confusion_matrix(y_test,y_test_pred),annot=True, fmt='d',cbar=False, cmap='rainbow')
plt.xlabel('Predicted Label')
plt.ylabel('Actual Label')
plt.title('Confusion Matrix')
plt.show()


# In[37]:


#classification report of test data
print(classification_report(y_test,y_test_pred))


# In[ ]:




