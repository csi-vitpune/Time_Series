#!/usr/bin/env python
# coding: utf-8

# ## Importing all the necessary libraries required

# In[ ]:


# Numpy helps in performing all the mathematical operations
# pandas helps in rading the data 
# Matplotlib helps in plotting the graphs
# Helps in splitting the data traing dataset and testing dataset
# Pairplots helps in deciding the model


# In[3]:


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


# ## Steps
# 
# 1. Reading the data
# 
# 2. Eliminating the unnecessary columns
# 
# 3. Check if we've some `Nan` Values
# 
# * Makking the patterns(Different graphs)
# 
# 4. Make some pairlots 
# 
# 5. Based on pairplots choose a model
# 
# 6. Check the accuracy
# 
# 7. If the accuracy is low use Hyperparametrs to tune the model and improve the accuracy
# 

# ### Importing the data

# In[4]:


data=pd.read_csv("C:/Users/User/Desktop/Capstone 2/NESTLEIND.csv")


# ### Reading the first few lines of the data

# In[5]:


data.head()


# In[6]:


data.tail(10)


# In[5]:


# Getting the shape of the data
data.shape


# ### Removing Some Unnecesary Columns like:
# 
# 1. Date
# 
# 2. Symbol
# 
# 3. Series
# 
# 4. Turnover

# In[8]:


data.drop("Date",axis=1,inplace=True)
data.drop("Symbol",axis=1,inplace=True)
data.drop("Series",axis=1,inplace=True)
data.drop("Turnover",axis=1,inplace=True)


# In[10]:


# Checking Whether Unnecessarybcolumns are removed,Storing the data into a different variable

final_data=data
final_data.head(10)


# In[8]:


# Above we're Done with removing the unnecessary columns


# ### Let's Check if we've Some `Nan`  Values
# 
# Suppose if the data contains more than 50% of `Nan` values then drop it
# else replace it with mean value or median value
# 

# In[9]:


final_data.isna().sum()


# ## Above we've got only 350 `Nan` values in the `Trades` column 
# 
# * Let's Either remove the entire column or replace them with the most repeated values

# In[10]:


data.info()


# In[11]:


data.describe()


# ### As we're required to find the closing price let's find the realtion of `Trades` and `Close`

# In[12]:


fig,ax=plt.subplots(figsize=(15,10))
ax.scatter(data['Trades'],data['Close']);
ax.set(title="Trades Vs Close",
      xlabel='Trades',
      ylabel='Close');


# In[13]:


# Let's Finally drop the `Trades`  Column


# In[14]:


final_data.drop('Trades',axis=1,inplace=True)


# In[15]:


final_data.head()


# ### Let's make a correlatoioinal Matrix 

# In[16]:


corr_mat=final_data.corr()
corr_mat


# ### Let's make it more readable and pretier by plotting it's heatmap

# In[17]:


fig,ax=plt.subplots(figsize=(15,10))
ax=sns.heatmap(corr_mat,annot=True,linewidths=0.5,cmap="YlGnBu")


# ### As above we've got many Dark blue Squares Which shows they are highly correlated with each other
# 
# 1. Let's give labels to X and y
# 
# 2. Split the data
# 
# 3. Let's Check which are the highly correlated columns.
# 
# 4. Remove some of them as each will be performing the same thing.

# In[18]:


X=final_data.drop('Close',axis=1)
y=final_data['Close']


# In[19]:


X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2)


# In[20]:


def correlation(dataset,threshold):
    col_corr=set()
    corr_matrix=final_data.corr()
    for i in range(len(corr_matrix.columns)):
        for j in range(i):
            if abs(corr_matrix.iloc[i,j])>threshold:
                col_name=corr_matrix.columns[i]
                col_corr.add(col_name)
            
        
    return col_corr


# In[21]:


corr_features=correlation(X_train,0.8)
corr_features


# In[22]:


final_data.drop('High',axis=1,inplace=True)
final_data.drop('Last',axis=1,inplace=True)
final_data.drop('Open',axis=1,inplace=True)


# ### Storing the remaining data into a different variable

# In[23]:


remaining_data=final_data


# In[24]:


remaining_data.head(10)


# ### Above we've got our final dataset
# 
# Now let's
# 
# 1. Split the data again as we've some new one now.
# 
# 2. Make a pairplot for selecting the model

# In[25]:


X=remaining_data.drop('Close',axis=1)
y=remaining_data['Close']


# In[26]:


X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2)


# In[27]:


sns.pairplot(corr_mat);


# #### Above we've got  a perfectly splitted data so lets use either `LinearRegression()` or `RandomForestRegressor()`

# In[41]:


np.random.seed(42)
model=LinearRegression()
model.fit(X_train,y_train)


# In[42]:


model.score(X_test,y_test)*100


# In[30]:


y_preds=model.predict(X_test)
y_preds


# In[31]:


remaining_data.head()


# In[43]:


model.predict([[107,107,107,40,40,20]])[0]


# ### Let's make a pickle file

# In[33]:


import pickle
pickle.dump(model,open("Nestle_predictor.pkl","wb"))


# In[ ]:




