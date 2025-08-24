#!/usr/bin/env python
# coding: utf-8

# In[92]:


import pandas as pd
import numpy as np
import sklearn.preprocessing as pp
import datetime as dt
from sklearn.cluster import KMeans

import matplotlib.pyplot as plt
import seaborn as sbn


# ## Read CSV File

# In[93]:


df = pd.read_csv('../exports/online_retail_cleaningdata.csv')


# In[94]:


df.head()


# In[95]:


df.info()


# In[96]:


df['date'] = pd.to_datetime(df['date'])


# In[97]:


df.info()


# In[98]:


last_date = df['date'].max()
last_date


# In[99]:


first_date = df['date'].min()
first_date


# In[100]:


three_months_ago = last_date - pd.DateOffset(months=3)
three_months_ago


# In[101]:


df_last_3month = df[df['date'] >= three_months_ago]
df_last_3month


# In[ ]:





# In[102]:


rfm_last_3month = (df_last_3month.groupby('customer_id').agg(
                                        recency = ('date' , lambda x: (last_date - x.max()).days),
                                        frequency = ('invoice_id' , 'nunique'),
                                        monetary = ('sales' , 'sum')).reset_index())


# In[103]:


rfm_last_3month


# ## **CLV = AOV * Purchase Frequency * Customer Lifespan**

# 
# **AOV (Average Order Value)**
# 

# In[104]:


rfm_last_3month['aov'] = (rfm_last_3month['monetary'] / rfm_last_3month['frequency']).clip(lower=1)
rfm_last_3month


# In[105]:


rfm_last_3month['purchase_freq'] = rfm_last_3month['frequency']
rfm_last_3month


# ## Lifespan per Months

# In[106]:


rfm_last_3month['lifespn_per_month'] = (rfm_last_3month['recency'] / 30).clip(lower=1)
rfm_last_3month


# ## Lifespan = The Mean of All Recencies

# In[107]:


avg_recency = rfm_last_3month['recency'].mean()


# In[108]:


avg_lifespn = avg_recency / 30


# In[109]:


rfm_last_3month['lifespn_avg'] = avg_lifespn
rfm_last_3month


# In[110]:


rfm_last_3month['clv_lifesp_m'] = (
                            rfm_last_3month['aov'] *
                            rfm_last_3month['purchase_freq']*
                            rfm_last_3month['lifespn_per_month']
)


# In[111]:


rfm_last_3month['clv_lifesp_avg'] = (
                            rfm_last_3month['aov'] *
                            rfm_last_3month['purchase_freq']*
                            rfm_last_3month['lifespn_avg']
)


# In[112]:


rfm_last_3month


# In[117]:


plt.figure(figsize=(18,8))

# Recency vs Frequency
plt.subplot(2, 3, 1)
plt.scatter(rfm_last_3month['recency'], rfm_last_3month['frequency'], color='green', alpha=0.5)
plt.xlabel('Recency')
plt.ylabel('Frequency')
plt.title('Recency vs Frequency')

# Recency vs Monetary
plt.subplot(2, 3, 2)
plt.scatter(rfm_last_3month['recency'], rfm_last_3month['monetary'], color='blue', alpha=0.5)
plt.xlabel('Recency')
plt.ylabel('Monetary')
plt.title('Recency vs Monetary')

# Frequency vs Monetary
plt.subplot(2, 3, 3)
plt.scatter(rfm_last_3month['frequency'], rfm_last_3month['monetary'], color='red', alpha=0.5)
plt.xlabel('Frequency')
plt.ylabel('Monetary')
plt.title('Frequency vs Monetary')

# recency vs clv_lifesp_avg
plt.subplot(2, 3, 4)
plt.scatter(rfm_last_3month['clv_lifesp_avg'], rfm_last_3month['recency'], color='green', alpha=0.6)
plt.xlabel('clv_lifesp_avg')
plt.ylabel('recency')
plt.title('Scatter Plot - recency vs clv_lifesp_avg')

# Frequency vs clv_lifesp_avg
plt.subplot(2, 3, 5)
plt.scatter(rfm_last_3month['clv_lifesp_avg'], rfm_last_3month['frequency'], color='blue', alpha=0.6)
plt.xlabel('clv_lifesp_avg')
plt.ylabel('Frequency')
plt.title('Scatter Plot - Frequency vs clv_lifesp_avg')

# Frequency vs clv_lifesp_avg
plt.subplot(2, 3, 6)
plt.scatter(rfm_last_3month['clv_lifesp_avg'], rfm_last_3month['monetary'], color='red', alpha=0.6)
plt.xlabel('clv_lifesp_avg')
plt.ylabel('Monetary')
plt.title('Scatter Plot - Monetary vs clv_lifesp_avg')



plt.tight_layout()
plt.show()


# ## Normalization,Standardization

# In[118]:


relevant_cols = ['recency' , 'frequency' , 'monetary' , 'clv_lifesp_m' , 'clv_lifesp_avg']


# In[119]:


rfm_clv = rfm_last_3month[relevant_cols]
rfm_clv.head()


# ### MinMax Normalization

# In[120]:


scaler1 = pp.MinMaxScaler()
minmax_sclr = scaler1.fit_transform(rfm_clv)


# In[121]:


minmax_sclr


# In[122]:


# Array to DataFrame

minmax_df = pd.DataFrame(minmax_sclr, columns=relevant_cols)
minmax_df.head()


# In[123]:


plt.figure(figsize=(18,8))

# Recency vs Frequency
plt.subplot(2, 3, 1)
plt.scatter(minmax_df['recency'], minmax_df['frequency'], color='green', alpha=0.5)
plt.xlabel('Recency')
plt.ylabel('Frequency')
plt.title('Recency vs Frequency')

# Recency vs Monetary
plt.subplot(2, 3, 2)
plt.scatter(minmax_df['recency'], minmax_df['monetary'], color='blue', alpha=0.5)
plt.xlabel('Recency')
plt.ylabel('Monetary')
plt.title('Recency vs Monetary')

# Frequency vs Monetary
plt.subplot(2, 3, 3)
plt.scatter(minmax_df['frequency'], minmax_df['monetary'], color='red', alpha=0.5)
plt.xlabel('Frequency')
plt.ylabel('Monetary')
plt.title('Frequency vs Monetary')

# recency vs clv_lifesp_avg
plt.subplot(2, 3, 4)
plt.scatter(minmax_df['clv_lifesp_avg'], minmax_df['recency'], color='green', alpha=0.6)
plt.xlabel('clv_lifesp_avg')
plt.ylabel('recency')
plt.title('Scatter Plot - recency vs clv_lifesp_avg')

# Frequency vs clv_lifesp_avg
plt.subplot(2, 3, 5)
plt.scatter(minmax_df['clv_lifesp_avg'], minmax_df['frequency'], color='blue', alpha=0.6)
plt.xlabel('clv_lifesp_avg')
plt.ylabel('Frequency')
plt.title('Scatter Plot - Frequency vs clv_lifesp_avg')

# Frequency vs clv_lifesp_avg
plt.subplot(2, 3, 6)
plt.scatter(minmax_df['clv_lifesp_avg'], minmax_df['monetary'], color='red', alpha=0.6)
plt.xlabel('clv_lifesp_avg')
plt.ylabel('Monetary')
plt.title('Scatter Plot - Monetary vs clv_lifesp_avg')



plt.tight_layout()
plt.show()


# ### Standard Normalization

# In[124]:


scaler2 = pp.StandardScaler()
std_sclr = scaler2.fit_transform(rfm_clv)


# In[125]:


std_sclr


# In[126]:


# Array to DataFrame

std_df = pd.DataFrame(std_sclr , columns=relevant_cols)
std_df.head()


# In[127]:


plt.figure(figsize=(18,8))

# Recency vs Frequency
plt.subplot(2, 3, 1)
plt.scatter(std_df['recency'], std_df['frequency'], color='green', alpha=0.5)
plt.xlabel('Recency')
plt.ylabel('Frequency')
plt.title('Recency vs Frequency')

# Recency vs Monetary
plt.subplot(2, 3, 2)
plt.scatter(std_df['recency'], std_df['monetary'], color='blue', alpha=0.5)
plt.xlabel('Recency')
plt.ylabel('Monetary')
plt.title('Recency vs Monetary')

# Frequency vs Monetary
plt.subplot(2, 3, 3)
plt.scatter(std_df['frequency'], std_df['monetary'], color='red', alpha=0.5)
plt.xlabel('Frequency')
plt.ylabel('Monetary')
plt.title('Frequency vs Monetary')

# recency vs clv_lifesp_avg
plt.subplot(2, 3, 4)
plt.scatter(std_df['clv_lifesp_avg'], std_df['recency'], color='green', alpha=0.6)
plt.xlabel('clv_lifesp_avg')
plt.ylabel('recency')
plt.title('Scatter Plot - recency vs clv_lifesp_avg')

# Frequency vs clv_lifesp_avg
plt.subplot(2, 3, 5)
plt.scatter(std_df['clv_lifesp_avg'], std_df['frequency'], color='blue', alpha=0.6)
plt.xlabel('clv_lifesp_avg')
plt.ylabel('Frequency')
plt.title('Scatter Plot - Frequency vs clv_lifesp_avg')

# Frequency vs clv_lifesp_avg
plt.subplot(2, 3, 6)
plt.scatter(std_df['clv_lifesp_avg'], std_df['monetary'], color='red', alpha=0.6)
plt.xlabel('clv_lifesp_avg')
plt.ylabel('Monetary')
plt.title('Scatter Plot - Monetary vs clv_lifesp_avg')



plt.tight_layout()
plt.show()


# ### Robust Normalization

# In[128]:


scaler3 = pp.RobustScaler()
robust_sclr = scaler3.fit_transform(rfm_clv)


# In[129]:


robust_sclr


# In[130]:


# Array to DataFrame

robust_df = pd.DataFrame(robust_sclr , columns=relevant_cols)
robust_df.head()


# In[131]:


plt.figure(figsize=(18,8))

# Recency vs Frequency
plt.subplot(2, 3, 1)
plt.scatter(robust_df['recency'], robust_df['frequency'], color='green', alpha=0.5)
plt.xlabel('Recency')
plt.ylabel('Frequency')
plt.title('Recency vs Frequency')

# Recency vs Monetary
plt.subplot(2, 3, 2)
plt.scatter(robust_df['recency'], robust_df['monetary'], color='blue', alpha=0.5)
plt.xlabel('Recency')
plt.ylabel('Monetary')
plt.title('Recency vs Monetary')

# Frequency vs Monetary
plt.subplot(2, 3, 3)
plt.scatter(robust_df['frequency'], robust_df['monetary'], color='red', alpha=0.5)
plt.xlabel('Frequency')
plt.ylabel('Monetary')
plt.title('Frequency vs Monetary')

# recency vs clv_lifesp_avg
plt.subplot(2, 3, 4)
plt.scatter(robust_df['clv_lifesp_avg'], robust_df['recency'], color='green', alpha=0.6)
plt.xlabel('clv_lifesp_avg')
plt.ylabel('recency')
plt.title('Scatter Plot - recency vs clv_lifesp_avg')

# Frequency vs clv_lifesp_avg
plt.subplot(2, 3, 5)
plt.scatter(robust_df['clv_lifesp_avg'], robust_df['frequency'], color='blue', alpha=0.6)
plt.xlabel('clv_lifesp_avg')
plt.ylabel('Frequency')
plt.title('Scatter Plot - Frequency vs clv_lifesp_avg')

# Frequency vs clv_lifesp_avg
plt.subplot(2, 3, 6)
plt.scatter(robust_df['clv_lifesp_avg'], robust_df['monetary'], color='red', alpha=0.6)
plt.xlabel('clv_lifesp_avg')
plt.ylabel('Monetary')
plt.title('Scatter Plot - Monetary vs clv_lifesp_avg')



plt.tight_layout()
plt.show()


# ## Clustering - KMeans - ElbowPlot

# In[132]:


def find_best_clusters(df, maximum_K):

    clusters_centers = []
    k_values = []

    for k in range(1, maximum_K):

        kmeans_model = KMeans(n_clusters = k)
        kmeans_model.fit(df)

        clusters_centers.append(kmeans_model.inertia_)
        k_values.append(k)


    return clusters_centers, k_values


# In[133]:


def generate_elbow_plot(clusters_centers, k_values):

    figure = plt.subplots(figsize = (12, 6))
    plt.plot(k_values, clusters_centers, 'o-', color = 'orange')
    plt.xlabel("Number of Clusters (K)")
    plt.ylabel("Cluster Inertia")
    plt.title("Elbow Plot of KMeans")
    plt.show()


# In[134]:


clusters_centers, k_values = find_best_clusters(std_sclr, 18)

generate_elbow_plot(clusters_centers, k_values)


# In[135]:


kmeans_model = KMeans(n_clusters = 5)
kmeans_model.fit(std_sclr)


# In[136]:


std_df["clusters"] = kmeans_model.labels_
std_df.head()


# In[138]:


rfm_last_3month['clusters'] = kmeans_model.labels_
rfm_last_3month


# In[139]:


plt.figure(figsize=(18,8))

# Recency vs Frequency
plt.subplot(2, 3, 1)
plt.scatter(std_df['recency'], std_df['frequency'],c = std_df['clusters'] , alpha=0.5)
plt.xlabel('Recency')
plt.ylabel('Frequency')
plt.title('Recency vs Frequency')

# Recency vs Monetary
plt.subplot(2, 3, 2)
plt.scatter(std_df['recency'], std_df['monetary'], c = std_df['clusters'], alpha=0.5)
plt.xlabel('Recency')
plt.ylabel('Monetary')
plt.title('Recency vs Monetary')

# Frequency vs Monetary
plt.subplot(2, 3, 3)
plt.scatter(std_df['frequency'], std_df['monetary'], c = std_df['clusters'], alpha=0.5)
plt.xlabel('Frequency')
plt.ylabel('Monetary')
plt.title('Frequency vs Monetary')

# recency vs clv_lifesp_avg
plt.subplot(2, 3, 4)
plt.scatter(std_df['clv_lifesp_avg'], std_df['recency'], c = std_df['clusters'] , alpha=0.6)
plt.xlabel('clv_lifesp_avg')
plt.ylabel('recency')
plt.title('Scatter Plot - recency vs clv_lifesp_avg')

# Frequency vs clv_lifesp_avg
plt.subplot(2, 3, 5)
plt.scatter(std_df['clv_lifesp_avg'], std_df['frequency'], c = std_df['clusters'] , alpha=0.6)
plt.xlabel('clv_lifesp_avg')
plt.ylabel('Frequency')
plt.title('Scatter Plot - Frequency vs clv_lifesp_avg')

# Frequency vs clv_lifesp_avg
plt.subplot(2, 3, 6)
plt.scatter(std_df['clv_lifesp_avg'], std_df['monetary'], c = std_df['clusters'] , alpha=0.6)
plt.xlabel('clv_lifesp_avg')
plt.ylabel('Monetary')
plt.title('Scatter Plot - Monetary vs clv_lifesp_avg')



plt.tight_layout()
plt.show()


# In[141]:


fig, axes = plt.subplots(1, 2, figsize=(14,6))  # 1 ردیف، 2 ستون

sbn.stripplot(x='clusters', y='recency' , data = rfm_last_3month ,  ax=axes[0])
axes[0].set_title("Recency by Clusters - rfm_last_3month")


sbn.stripplot(x='clusters', y='recency' , data = std_df ,  ax=axes[1])
axes[1].set_title("Recency by Clusters - std_df")


plt.tight_layout()
plt.show()


# In[142]:


fig, axes = plt.subplots(1, 2, figsize=(14,6))  # 1 ردیف، 2 ستون

sbn.stripplot(x='clusters', y='frequency' , data = rfm_last_3month ,  ax=axes[0])
axes[0].set_title("Recency by Frequency - rfm_last_3month")


sbn.stripplot(x='clusters', y='frequency' , data = std_df ,  ax=axes[1])
axes[1].set_title("Recency by Frequency - std_df")


plt.tight_layout()
plt.show()


# In[143]:


fig, axes = plt.subplots(1, 2, figsize=(14,6))  # 1 ردیف، 2 ستون

sbn.stripplot(x='clusters', y='monetary' , data = rfm_last_3month ,  ax=axes[0])
axes[0].set_title("Recency by Monetary - rfm_last_3month")


sbn.stripplot(x='clusters', y='monetary' , data = std_df ,  ax=axes[1])
axes[1].set_title("Recency by Monetary - std_df")


plt.tight_layout()
plt.show()


# In[144]:


fig, axes = plt.subplots(1, 2, figsize=(14,6))  # 1 ردیف، 2 ستون

sbn.stripplot(x='clusters', y='clv_lifesp_avg' , data = rfm_last_3month ,  ax=axes[0])
axes[0].set_title("Recency by CLV - rfm_clv")


sbn.stripplot(x='clusters', y='clv_lifesp_avg' , data = std_df ,  ax=axes[1])
axes[1].set_title("Recency by CLV - std_df")


plt.tight_layout()
plt.show()


# In[145]:


std_df['customer_id'] = rfm_last_3month['customer_id'].values
std_df.head()


# In[146]:


clusters_mean = std_df.groupby('clusters').agg(
                                                avg_recency = ('recency' , 'mean'),
                                                avg_frequency = ('frequency','mean'),
                                                avg_monetary = ('monetary','mean'),
                                                avg_clv_lifesp_m = ('clv_lifesp_m','mean'),
                                                avg_clv_lifesp_avg = ('clv_lifesp_avg','mean'),
                                                customers_count = ('customer_id' , 'count')

).reset_index()


# In[147]:


clusters_mean


# In[85]:


#rfm_clv.to_excel('../exports/rfm_clv_3month_realdata.xlsx')


# In[86]:


#rfm_clv.to_csv('../exports/rfm_clv_3month_realdata.csv')


# In[148]:


std_df.to_csv('../exports/rfm_clv_normalized_kmeans_3month.csv')


# In[149]:


std_df.to_excel('../exports/rfm_clv_normalized_kmeans_3month.xlsx')


# In[150]:


clusters_mean.to_csv('../exports/rfm_clv_3month_mean.csv')


# In[151]:


clusters_mean.to_excel('../exports/rfm_clv_3month_mean.xlsx')


# In[152]:


rfm_last_3month.to_csv('../exports/rfm_clv_3month_realdata.csv')


# In[153]:


rfm_last_3month.to_excel('../exports/rfm_clv_3month_realdata.xlsx')


# In[ ]:




