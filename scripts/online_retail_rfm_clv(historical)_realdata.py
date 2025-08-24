#!/usr/bin/env python
# coding: utf-8

# In[25]:


import os
import pandas as pd
import numpy as np

#Data Visualization
import matplotlib.pyplot as plt
import plotly.express as px
import seaborn as sbn


# In[26]:


df = pd.read_csv('../exports/online_retail_cleaningdata.csv')


# In[27]:


df.head()


# In[28]:


df.isnull().sum()


# # Finding RFM values

# ## Recency

# In[29]:


day = pd.to_datetime("2011-12-10").normalize()
df['date'] = pd.to_datetime(df['date']).dt.normalize()


# In[30]:


df['date'].max() , df['date'].min()


# In[31]:


recency = df.groupby(['customer_id']).agg({"date": lambda x:((day-x.max()).days)}).reset_index()
recency = recency.rename(columns={'date': 'recency'})
recency


# In[32]:


df.customer_id.unique().shape


# ## Ferequency (In Two Ways)

# In[33]:


frequency_1 = df.drop_duplicates(subset ='invoice_id').groupby(['customer_id'])[['invoice_id']].count()
frequency_1


# In[34]:


frequency = df.groupby(['customer_id'])[['invoice_id']].nunique().reset_index()
frequency = frequency.rename(columns={'invoice_id' : 'frequency'})
frequency.head()


# ## Monetary

# In[35]:


monetary = df.groupby(['customer_id'])[['sales']].sum().reset_index()
monetary = monetary.rename(columns={'sales' : 'monetary'})
monetary.head()


# ## RFM Values

# In[36]:


rfm_values = recency.merge(frequency, on='customer_id')
rfm_values = rfm_values.merge(monetary, on='customer_id')

rfm_values.head()


# In[37]:


rfm_values.isnull().sum()


# In[38]:


rfm_sorted = rfm_values.sort_values('monetary' , ascending=False)
rfm_sorted.head()


# ## CLV Historical

# In[39]:


df.head()


# In[40]:


clv_historical = df.groupby('customer_id')[['sales']].sum().reset_index()
clv_historical.columns = ['customer_id', 'clv_historical']
clv_historical.sort_values(by = 'clv_historical' , ascending = False)


# In[41]:


rfm_clv = rfm_values.merge(clv_historical , on = 'customer_id' , how = 'left')
rfm_clv.head()


# In[42]:


relevant_cols = ["recency", "frequency", "monetary" , "clv_historical"]
rfm_clv_n = rfm_clv[relevant_cols]


# In[43]:


rfm_clv_n.head()


# In[44]:


plt.figure(figsize=(18,8))

# Recency vs Frequency
plt.subplot(2, 3, 1)
plt.scatter(rfm_clv_n['recency'], rfm_clv_n['frequency'], color='green', alpha=0.5)
plt.xlabel('Recency')
plt.ylabel('Frequency')
plt.title('Recency vs Frequency')

# Recency vs Monetary
plt.subplot(2, 3, 2)
plt.scatter(rfm_clv_n['recency'], rfm_clv_n['monetary'], color='blue', alpha=0.5)
plt.xlabel('Recency')
plt.ylabel('Monetary')
plt.title('Recency vs Monetary')

# Frequency vs Monetary
plt.subplot(2, 3, 3)
plt.scatter(rfm_clv_n['frequency'], rfm_clv_n['monetary'], color='red', alpha=0.5)
plt.xlabel('Frequency')
plt.ylabel('Monetary')
plt.title('Frequency vs Monetary')

# recency vs clv_historical
plt.subplot(2, 3, 4)
plt.scatter(rfm_clv_n['recency'], rfm_clv_n['clv_historical'], color='green', alpha=0.6)
plt.xlabel('recency')
plt.ylabel('clv_historical')
plt.title('Scatter Plot - recency vs clv_historical')

# Frequency vs clv_historical
plt.subplot(2, 3, 5)
plt.scatter(rfm_clv_n['frequency'], rfm_clv_n['clv_historical'], color='blue', alpha=0.6)
plt.xlabel('Frequency')
plt.ylabel('clv_historical')
plt.title('Scatter Plot - Frequency vs clv_historical')

# Frequency vs clv_historical
plt.subplot(2, 3, 6)
plt.scatter(rfm_clv_n['monetary'], rfm_clv_n['clv_historical'], color='red', alpha=0.6)
plt.xlabel('Monetary')
plt.ylabel('clv_historical')
plt.title('Scatter Plot - Monetary vs clv_historical')


plt.tight_layout()
plt.show()


# In[ ]:





# In[45]:


rfm_clv.to_csv('../exports/online_retail_rfm_clv(historical)_realdata.csv')


# In[46]:


rfm_clv.to_excel('../exports/online_retail_rfm_clv(historical)_realdata.xlsx')


# In[48]:


rfm_clv


# In[ ]:




