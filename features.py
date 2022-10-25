#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression


# In[2]:


data=pd.read_csv(r"C:\Users\Hrutika\Desktop\Software Project\student-mat.csv")


# In[3]:


data


# In[4]:


plt.figure(figsize=(14,12))
sns.heatmap(data.corr(), annot=True)
plt.show()


# In[5]:


data.isnull().sum()


# In[6]:


data.dtypes


# In[7]:


nonnumeric_columns=[data.columns[index] for index, dtype in enumerate(data.dtypes) if dtype=='object']
nonnumeric_columns


# In[8]:


for column in nonnumeric_columns:
    print(f'{column}: {data[column].unique()}')


# In[9]:


data['Mjob']= data['Mjob'].apply(lambda x:'m_' + x)
data['Fjob']= data['Fjob'].apply(lambda x:'f_' + x)
data['reason']= data['reason'].apply(lambda x:'r_' + x)
data['guardian']= data['guardian'].apply(lambda x:'g_' + x)


# In[10]:


data


# In[11]:


dummies = pd.concat([pd.get_dummies(data['Mjob']),pd.get_dummies(data['Fjob']),pd.get_dummies(data['reason']),pd.get_dummies(data['guardian'])],axis=1)


# In[12]:


dummies


# In[13]:


data = pd.concat([data,dummies], axis=1)
data.drop(['Mjob','Fjob','reason','guardian'], axis=1, inplace=True)
data


# In[14]:


nonnumeric_columns=[data.columns[index] for index, dtype in enumerate(data.dtypes) if dtype=='object']
for column in nonnumeric_columns:
    print(f'{column}: {data[column].unique()}')


# In[15]:


encoder = LabelEncoder()
for column in nonnumeric_columns:
    data[column] = encoder.fit_transform(data[column])
    


# In[16]:


data.dtypes


# In[17]:


for dtype in data.dtypes:
    print(dtype)


# In[18]:


y=data['G3']
x=data.drop('G3',axis=1)


# In[19]:


y


# In[20]:


x


# In[24]:


scalar = StandardScaler()
x = pd.DataFrame(scalar.fit_transform(x),columns=x.columns)


# In[25]:


x


# In[27]:


X_train, X_test, y_train, y_test = train_test_split(x,y,train_size=0.7)


# In[28]:


model=LinearRegression()
model.fit(X_train,y_train)


# In[29]:


print(f'Model R2: {model.score(X_test,y_test)}')


# In[ ]:




