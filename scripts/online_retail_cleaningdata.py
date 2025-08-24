#!/usr/bin/env python
# coding: utf-8

# In[76]:


import os
import pandas as pd
import numpy as np
import datetime as dt

#Data Visualization
import matplotlib.pyplot as plt
import plotly.express as px


# In[77]:


df = pd.read_excel('../data/OnlineRetail.xlsx')


# In[78]:


df.head()


# In[79]:


df.columns


# In[80]:


df.shape


# In[81]:


df.info()


# In[82]:


df.info(memory_usage='deep')


# In[83]:


df.columns=['invoice_id', 'item_id', 'description', 'quantity', 'date',
       'unit_price', 'customer_id', 'country']


# In[84]:


df.columns


# In[85]:


#Number of invoice_id
df.invoice_id.unique().shape


# In[86]:


df.invoice_id.unique()


# In[87]:


df.date.min() , df.date.max()


# In[88]:


df['date'] = df['date'].apply(lambda x:dt.datetime(x.year , x.month , x.day))


# In[89]:


df.head()


# In[90]:


df.country.unique().shape


# In[91]:


df.customer_id.unique().shape


# In[92]:


df.customer_id.isnull().sum()


# In[93]:


df.isnull().sum()


# In[94]:


df.isnull().sum().sum()


# In[95]:


#Create Sales column
df['sales'] = df['quantity'] * df['unit_price']


# In[96]:


df.head()


# In[97]:


df_describe = df.filter(['quantity', 'sales'])
df_describe.describe()


# In[98]:


df_describe.corr()


# In[99]:


df.sales.quantile([0,0.25,0.5,0.75,1])


# In[100]:


df.quantity.quantile([0,0.25,0.5,0.75,1])


# In[101]:


cals = list(df.columns)
df = df[cals[0:6]+[cals[-1]]+cals[6:8]]


# In[102]:


df.head()


# In[103]:


df[df['customer_id'].isnull()]


# In[104]:


df['customer_id'].isnull().sum()


# In[105]:


#Delete rows that customer_id is NULL
df = df.dropna(subset=['customer_id'])


# In[106]:


df['customer_id'].isnull().sum()


# ## Invoice IDs that start with C

# In[107]:


df[df['invoice_id'].str.startswith('C' , na=False)]


# In[108]:


df[df['invoice_id'].str.startswith('C' , na=False)].shape


# In[109]:


#Delete rows that Invoice Id start with C
df = df[~df['invoice_id'].str.startswith('C' , na=False)]


# In[110]:


df.shape


# ## unit_price<0 & quantity<0

# In[111]:


df[df['unit_price']<0].shape


# In[112]:


df[df['quantity']<0].shape


# In[113]:


df[df['sales'] < 0 ]


# In[114]:


df.shape


# ## The 99th percentile represents the top 1% of the data, which are usually outliers.

# In[115]:


df['unit_price'].max() , df['unit_price'].min()


# In[116]:


df['quantity'].max() , df['quantity'].min()


# In[117]:


quantity_99 = df['quantity'].quantile(0.99)
quantity_99


# In[118]:


unit_price_99=df['unit_price'].quantile(0.99)
unit_price_99


# In[119]:


df[(df['unit_price'] <= unit_price_99) ].shape


# In[120]:


df[(df['unit_price'] > unit_price_99) ].shape


# In[121]:


df = df[df['unit_price'] <= unit_price_99]


# In[122]:


df[df['quantity'] <= quantity_99].shape


# In[123]:


df[df['quantity'] > quantity_99].shape


# In[124]:


df = df[df['quantity'] <= quantity_99 ]


# In[125]:


df.shape


# In[126]:


df.head()


# In[127]:


df.to_csv('../exports/online_retail_cleaningdata.csv')


# In[128]:


df.to_excel('../exports/online_retail_cleaningdata.xlsx')


# In[129]:


cus_sum=df.groupby(['customer_id'])['sales'].sum().reset_index(name='sales_sum')
cus_sum.head()


# In[130]:


df.country.value_counts()


# In[131]:


df.groupby('customer_id').size().reset_index(name='count')


# In[132]:


df.dtypes


# In[ ]:




