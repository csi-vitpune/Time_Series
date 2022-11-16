#!/usr/bin/env python
# coding: utf-8

# ## Importing the Necessary Libraries

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report,confusion_matrix
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
# from sklearn.metrics import plot_det_curve


# In[2]:


data=pd.read_csv("ADANIPORTS.NS.csv")


# In[3]:


data.head(10)


# ### Dropping the unnecessary Columns like:
#   1. Date

# In[4]:


data.drop('Date',axis=1,inplace=True)


# In[5]:


# Storing the left out data in a Different Variable `Final_data`


# In[6]:


final_data=data


# In[7]:


# Reading the first few lines of the data

final_data.head(10)


# ### Making Some patterns

# In[8]:


fig,ax=plt.subplots(figsize=(10,10))
ax.scatter(final_data['Open'],final_data['Close']);
ax.set(title="Open Vs Close",
      xlabel='Open',
      ylabel='Close');


# In[9]:


# Describind the entire data

final_data.describe()


# In[10]:


final_data.info()


# In[11]:


# Checkiong if there are any `Nan` values

final_data.isna().sum()


# In[12]:


# Let's fill all the `Nan` values with their respective column means

final_data['Open'].fillna(final_data['Open'].mean(),inplace=True)
final_data['High'].fillna(final_data['High'].mean(),inplace=True)
final_data['Low'].fillna(final_data['Low'].mean(),inplace=True)
final_data['Close'].fillna(final_data['Close'].mean(),inplace=True)
final_data['Adj Close'].fillna(final_data['Adj Close'].mean(),inplace=True)
final_data['Volume'].fillna(final_data['Volume'].mean(),inplace=True)


# #### Storing the filled data into a different variable `reamaining_data`

# In[13]:


remaining_data=final_data


# In[14]:


remaining_data.head(10)


# In[15]:


remaining_data.isna().sum()


# ### Making patterng between `Adj Close` and `Close`

# In[16]:


fig,ax=plt.subplots(figsize=(10,10))
ax.scatter(final_data['Adj Close'],final_data['Close']);
ax.set(title="Adj Close Vs Close",
      xlabel='Adj Close',
      ylabel='Close');


# ### Splitting the data into training and testing datasets

# In[17]:


X=remaining_data.drop('Close',axis=1)
y=remaining_data['Close']


# ## Making a Correlational Matrix

# In[18]:


corr_mat=remaining_data.corr()
corr_mat


# In[19]:


## Makinng the correlational matrix look more prettier

fig,ax=plt.subplots(figsize=(12,8))
ax=sns.heatmap(corr_mat,
              annot=True,
              linewidths=0.5,
              cmap='Blues',
              fmt='.2f')


# In[20]:


X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2)


# In[21]:


def correlation(dataset,threshold):
    col_corr=set()
    corr_matrix=final_data.corr()
    for i in range(len(corr_matrix.columns)):
        for j in range(i):
            if abs(corr_matrix.iloc[i,j])>threshold:
                col_name=corr_matrix.columns[i]
                col_corr.add(col_name)
            
        
    return col_corr


# In[22]:


corr_features=correlation(X_train,0.8)
corr_features


# ## Plotting pairplots

# In[23]:


sns.pairplot(corr_mat)


# In[24]:


np.random.seed(42)
model=RandomForestRegressor()
model.fit(X_train,y_train)


# In[25]:


model.score(X_test,y_test)


# In[26]:


y_preds=model.predict(X_test)
y_preds


# In[27]:


remaining_data.head(10)


# In[31]:


X_test[:10]


# In[32]:


model.predict([[669.400024,711.349976,667.150024,699.400146,18596690.0]])


# In[29]:


## Making a pickle file

import pickle
pickle.dump(model,open("Adani_predictor.pkl","wb"))


# In[ ]:




