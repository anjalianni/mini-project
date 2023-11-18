#!/usr/bin/env python
# coding: utf-8

# In[143]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn import metrics


# In[144]:


data_train=pd.read_csv('Train.csv')


# In[145]:


data_train.head()


# In[146]:


data_train.shape


# In[147]:


data_train.info()


# In[148]:


data_train.isnull().sum()


# In[149]:


data_train['Item_Weight'].mean()


# In[150]:


data_train['Item_Weight'].fillna(data_train['Item_Weight'].mean(),inplace = True)


# In[151]:


data_train.isnull().sum()


# In[152]:


mode_of_outlet_size = data_train.pivot_table(values='Outlet_Size',columns = 'Outlet_Type',aggfunc=(lambda x: x.mode()[0]))


# In[153]:


print(mode_of_outlet_size)


# In[154]:


missing_values = data_train['Outlet_Size'].isnull()


# In[155]:


print(missing_values)


# In[156]:


data_train.loc[missing_values,'Outlet_Size']=data_train.loc[missing_values,'Outlet_Type'].apply(lambda x: mode_of_outlet_size)


# In[157]:


data_train.isnull().sum()


# In[158]:


data_train.describe()


# In[159]:


sns.set()


# In[160]:


plt.figure(figsize=(6,6))
sns.distplot(data_train['Item_Weight'])
plt.show()


# In[161]:


plt.figure(figsize=(6,6))
sns.displot(data_train['Item_Visibility'])
plt.show()


# In[162]:


plt.figure(figsize=(6,6))
sns.displot(data_train['Item_Outlet_Sales'])
plt.show()


# In[163]:


plt.figure(figsize=(6,6))
sns.countplot(x='Outlet_Establishment_Year',data=data_train)
plt.show()


# In[164]:


plt.figure(figsize=(6,6))
sns.countplot(x='Item_Fat_Content',data=data_train)
plt.show()


# In[165]:


plt.figure(figsize=(6,6))
sns.countplot(x='Item_Type',data=data_train)
plt.show()


# In[166]:


plt.figure(figsize=(20,6))
sns.countplot(x='Item_Type',data=data_train)
plt.title('Item_Type count ')
plt.show()


# In[167]:


data_train.head()


# In[168]:


data_train['Item_Fat_Content'].value_counts()


# In[169]:


data_train.replace({'Item_Fat_Content':{'low fat':'Low Fat','LF':'Low Fat','reg':'Regular'}},inplace = True)


# In[170]:


data_train['Item_Fat_Content'].value_counts()


# In[171]:


encoder = LabelEncoder()


# In[172]:


data_train['Item_Identifier'] = encoder.fit_transform(data_train['Item_Identifier'])
data_train['Item_Fat_Content'] = encoder.fit_transform(data_train['Item_Fat_Content'])
data_train['Item_Type'] = encoder.fit_transform(data_train['Item_Type'])
data_train['Outlet_Identifier'] = encoder.fit_transform(data_train['Outlet_Identifier'])

data_train['Outlet_Location_Type'] = encoder.fit_transform(data_train['Outlet_Location_Type'])
data_train['Outlet_Type'] = encoder.fit_transform(data_train['Outlet_Type'])


# In[173]:


data_train['Outlet_Type'] = encoder.fit_transform(data_train['Outlet_Type'])


# In[174]:


data_train.head()


# In[181]:


a= data_train.drop(columns='Item_Outlet_Sales',axis=1)
y= data_train['Item_Outlet_Sales']
x=a.drop(columns='Outlet_Size',axis=1)


# In[182]:


print(x)


# In[183]:


print(y)


# In[184]:


x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=2)


# In[185]:


print(x.shape,x_train.shape,x_test.shape)


# In[186]:


from sklearn.linear_model import LinearRegression


# In[187]:


reg=LinearRegression()


# In[ ]:





# In[188]:


from xgboost import XGBRegressor


# In[ ]:





# In[189]:


regressor = XGBRegressor()


# In[190]:


regressor.fit(x_train,y_train)


# In[191]:


training_data_prediction = regressor.predict(x_train)


# In[193]:


r2_train = metrics.r2_score(y_train,training_data_prediction)


# In[194]:


print('R Squared value=',r2_train)


# In[195]:


test_data_prediction = regressor.predict(x_test)


# In[198]:


r2_test = metrics.r2_score(y_test,test_data_prediction)


# In[199]:


print('R Squared value=',r2_test)


# In[ ]:




