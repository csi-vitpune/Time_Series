#!/usr/bin/env python
# coding: utf-8

# In[32]:


import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

# ### Importong  the data

# In[33]:


data=pd.read_csv("KOTAKBANK.csv")


# ### Reading First few lines of data

# In[34]:


data.head(10)


# ### Dropping Unnecessary Columns Like:
#   * Date
#   * Symbol
#   * Series
#   * Turnover

# In[35]:


data.drop('Date',axis=1,inplace=True)
data.drop('Symbol',axis=1,inplace=True)
data.drop('Series',axis=1,inplace=True)
data.drop('Turnover',axis=1,inplace=True)


# #### Checking if there are any `Nan` values present in the dataset
# 
#    * if `Nan` values are 50% or above in a particular column let's simply remove the column
#    * if `Nan` values are less then just replace them with mean

# In[36]:


data.isna().sum()


# #### As we've got `Nan` values more than 50% in `Trades` column now let's remove that column

# In[37]:


data.drop('Trades',axis=1,inplace=True)


# In[38]:


# Getting the info of the dataset

data.info()


# In[39]:


# Getting the description of the data
data.describe()


# In[40]:


# Storing the data to a different variable
final_data=data


# #### As we've got some `Nan` values in the columns like:
#    1. Deliverable Volume
#    2.%Deliverble
# * Let's Fill them up with their individual column mean

# In[42]:


final_data['DeliverableVolume'].fillna(final_data['DeliverableVolume'].mean(),inplace=True)
final_data['Deliverble'].fillna(final_data['Deliverble'].mean(),inplace=True)


# In[43]:


# Reading the first few line of data
final_data.head()


# In[44]:


fig,ax=plt.subplots(figsize=(15,10))
ax.scatter(final_data['Volume'],final_data['Close'])
ax.set(title='Voliume Vs Close',
      xlabel='Open',
      ylabel='Close');


# #### Making a correlational matrix

# In[45]:


corr_mat=final_data.corr()
corr_mat


# #### Making the correlational matrix look more pretier by plotting it in a graph

# In[46]:


fig,ax=plt.subplots(figsize=(15,10))
ax=sns.heatmap(corr_mat,linewidths=.5,fmt='.2f',cmap='Blues',annot=True)


# ### Assigning and Splitting the data
# 

# In[47]:


X=final_data.drop('Close',axis=1)
y=final_data['Close']


# In[48]:


X_train,X_test,y_train,y_test=train_test_split(final_data,y,test_size=0.2)


# #### Printing the highly correlated values as each might give the same result

# In[49]:


def correlation(dataset,threshold):
    col_corr=set()
    corr_matrix=final_data.corr()
    for i in range(len(corr_matrix.columns)):
        for j in range(i):
            if abs(corr_matrix.iloc[i,j])>threshold:
                col_name=corr_matrix.columns[i]
                col_corr.add(col_name)
            
        
    return col_corr


# In[50]:


corr_features=correlation(X_train,0.8)
corr_features


# In[51]:


final_data.drop('High',axis=1,inplace=True)
final_data.drop('Last',axis=1,inplace=True)
final_data.drop('Open',axis=1,inplace=True)


# #### Storing the left out data in a different variable
# 

# In[52]:


remaining_data=final_data


# In[53]:


remaining_data.head(10)


# In[68]:


X=remaining_data.drop('Close',axis=1)
y=remaining_data['Close']


# In[69]:


X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2)


# #### Making pairplot of the data which will be helping us for selecting the model

# In[70]:


sns.pairplot(corr_mat)


# #### Using a model(RandomForestRegressor()) , fitting the data and Evaluating it.

# In[71]:


model=RandomForestRegressor(random_state=42)
model.fit(X_train,y_train)
model.score(X_test,y_test)*100


# ### Now let;s predict the data on some values

# In[72]:


y_preds=model.predict(X_test)
y_preds


# In[77]:


model.predict([[107,67,67,240,240,0.99]])[0]


# In[74]:


remaining_data.head()


# In[78]:


import pickle
kotakapp=open("KOTAKMODEL.pkl","wb")
pickle.dump(model,kotakapp)
kotakapp.close()


# In[ ]:




