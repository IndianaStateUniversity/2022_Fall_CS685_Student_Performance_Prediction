#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression


# In[2]:


given_url="http://bit.ly/w-data"
sample_data=pd.read_csv(given_url)
print("Retrieved dats: \n")
sample_data


# In[3]:


X=sample_data.iloc[:,:-1].values
Y=sample_data.iloc[:,1].values
print("Data converted into arrays")


# In[4]:


from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(X,Y,test_size=0.2,random_state=0)


# In[5]:


plt.scatter(X,Y)
plt.title("Hours vs Score")
plt.xlabel("Hours Studied")
plt.ylabel("Scores_Obtained")
plt.grid(5)
plt.show()


# In[6]:


lr_model=LinearRegression()
lr_model.fit(x_train,y_train)
print("Linear Regression Model Trained")


# In[7]:


Y_predicted=lr_model.predict(X)
plt.scatter(X,Y,label="Sample Data")
plt.plot(X,Y_predicted,color='red',label="Regression line")
plt.title("Hours vs Score")
plt.xlabel("Hours Studied")
plt.ylabel("Scores Obtained")
plt.legend("Sample Data")
plt.grid(5)
plt.show()


# In[8]:


eg_hours=[[9.25]]
eg_pred=lr_model.predict(eg_hours)
print("No of Hours ={}".format(eg_hours))
print("Predicted Score= {}".format(eg_pred[0]))


# In[9]:


#comparing actual vs predicted
y_pred=lr_model.predict(x_test)
corr_matrix=np.corrcoef(y_test,y_pred)
df = pd.DataFrame({'Actual':y_test,'Predicted':y_pred})
print(df)
corr=corr_matrix[0,1]
R_sq=corr**2
print("\n\n Coefficient of determination = ",R_sq)
print("\n\n thus our linear regression model can train :",R_sq*100,"% of data correctly")


# In[ ]:




