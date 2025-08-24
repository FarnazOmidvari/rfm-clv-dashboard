#!/usr/bin/env python
# coding: utf-8

# In[18]:


import os
import pandas as pd
import numpy as np
import datetime as dt

#Data Visualization
import matplotlib.pyplot as plt
import plotly.express as px


# In[19]:


df = pd.read_excel('../data/OnlineRetail.xlsx')


# In[20]:


df.head()


# In[21]:


df.columns


# In[22]:


df.shape


# In[23]:


df.info()


# In[24]:


df.info(memory_usage='deep')


# In[25]:


df.columns=['invoice_id', 'item_id', 'description', 'quantity', 'date',
       'unit_price', 'customer_id', 'country']


# In[26]:


df.columns


# In[27]:


#Number of invoice_id
df.invoice_id.unique().shape


# In[28]:


df.invoice_id.unique()


# In[29]:


df.date.min() , df.date.max()


# In[30]:


df['date'] = df['date'].apply(lambda x:dt.datetime(x.year , x.month , x.day))


# In[31]:


df.head()


# In[32]:


df.country.unique().shape


# In[33]:


df.customer_id.unique().shape


# In[34]:


df.customer_id.isnull().sum()


# In[35]:


df.isnull().sum()


# In[36]:


df.isnull().sum().sum()


# In[37]:


#Create Sales column
df['sales'] = df['quantity'] * df['unit_price']


# In[38]:


df.head()


# In[40]:


df_describe = df.filter(['quantity', 'sales'])
df_describe.describe()


# In[41]:


df_describe.corr()


# In[42]:


df.sales.quantile([0,0.25,0.5,0.75,1])


# In[43]:


df.quantity.quantile([0,0.25,0.5,0.75,1])


# In[44]:


cals = list(df.columns)
df = df[cals[0:6]+[cals[-1]]+cals[6:8]]


# In[45]:


df.head()


# In[46]:


df[df['customer_id'].isnull()]


# In[47]:


df['customer_id'].isnull().sum()


# In[48]:


#Delete rows that customer_id is NULL
df = df.dropna(subset=['customer_id'])


# In[49]:


df['customer_id'].isnull().sum()


# ## Invoice IDs that start with C

# In[50]:


df[df['invoice_id'].str.startswith('C' , na=False)]


# In[51]:


df[df['invoice_id'].str.startswith('C' , na=False)].shape


# In[52]:


#Delete rows that Invoice Id start with C
df = df[~df['invoice_id'].str.startswith('C' , na=False)]


# In[53]:


df.shape


# ## unit_price<0 & quantity<0

# In[54]:


df[df['unit_price']<0].shape


# In[55]:


df[df['quantity']<0].shape


# In[56]:


df[df['sales'] < 0 ]


# In[57]:


df.shape


# ## The 99th percentile represents the top 1% of the data, which are usually outliers.

# In[58]:


df['unit_price'].max() , df['unit_price'].min()


# In[59]:


df['quantity'].max() , df['quantity'].min()


# In[60]:


quantity_99 = df['quantity'].quantile(0.99)
quantity_99


# In[61]:


unit_price_99=df['unit_price'].quantile(0.99)
unit_price_99


# In[62]:


df[(df['unit_price'] <= unit_price_99) ].shape


# In[63]:


df[(df['unit_price'] > unit_price_99) ].shape


# In[64]:


df = df[df['unit_price'] <= unit_price_99]


# In[65]:


df[df['quantity'] <= quantity_99].shape


# In[66]:


df[df['quantity'] > quantity_99].shape


# In[67]:


df = df[df['quantity'] <= quantity_99 ]


# In[68]:


df.shape


# In[69]:


df.head()


# In[70]:


df.to_csv('../exports/online_retail_cleaningdata.csv')


# In[71]:


df.to_excel('../exports/online_retail_cleaningdata.xlsx')


# In[72]:


cus_sum=df.groupby(['customer_id'])['sales'].sum().reset_index(name='sales_sum')
cus_sum.head()


# In[73]:


df.country.value_counts()


# In[74]:


df.groupby('customer_id').size().reset_index(name='count')


# In[75]:


df.dtypes


# In[ ]:




