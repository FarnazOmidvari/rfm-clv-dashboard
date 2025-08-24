#!/usr/bin/env python
# coding: utf-8

# In[36]:


import pandas as pd
import numpy as np
import sklearn.preprocessing as pp


import matplotlib.pyplot as plt
import plotly.express as px
import seaborn as sbn


# In[37]:


rfm_clv = pd.read_csv('../exports/online_retail_rfm_clv(historical)_realdata.csv')
rfm_clv


# In[38]:


rfm_clv.info()


# In[39]:


relevant_cols = ["recency", "frequency", "monetary" , "clv_historical"]
rfm_clv_n = rfm_clv[relevant_cols]


# In[40]:


rfm_clv_n.head()


# In[ ]:





# 
#  # **Normalization,Standardization**

# In[41]:


import sklearn.preprocessing as pp


# ## **MinMax Normalization**

# In[42]:


scaler1 = pp.MinMaxScaler()
minmax_rfm = scaler1.fit_transform(rfm_clv_n)
minmax_rfm


# In[43]:


relevant_cols


# In[44]:


# Array → DataFrame

minmax_df = pd.DataFrame(minmax_rfm, columns=relevant_cols)
minmax_df.head()


# In[45]:


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

# recency vs clv_historical
plt.subplot(2, 3, 4)
plt.scatter(minmax_df['recency'], minmax_df['clv_historical'], color='green', alpha=0.6)
plt.xlabel('recency')
plt.ylabel('clv_historical')
plt.title('Scatter Plot - recency vs clv_historical')

# Frequency vs clv_historical
plt.subplot(2, 3, 5)
plt.scatter(minmax_df['frequency'], minmax_df['clv_historical'], color='blue', alpha=0.6)
plt.xlabel('Frequency')
plt.ylabel('clv_historical')
plt.title('Scatter Plot - Frequency vs clv_historical')

# Frequency vs clv_historical
plt.subplot(2, 3, 6)
plt.scatter(minmax_df['monetary'], minmax_df['clv_historical'], color='red', alpha=0.6)
plt.xlabel('Monetary')
plt.ylabel('clv_historical')
plt.title('Scatter Plot - Monetary vs clv_historical')



plt.tight_layout()
plt.show()


# ## **Standard Normalization**

# In[46]:


scaler2 = pp.StandardScaler()
std_rfm = scaler2.fit_transform(rfm_clv_n)
std_rfm


# In[47]:


# Array → DataFrame

std_df = pd.DataFrame(std_rfm , columns=relevant_cols)
std_df.head()


# In[48]:


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

# recency vs clv_historical
plt.subplot(2, 3, 4)
plt.scatter(std_df['recency'], std_df['clv_historical'],color='green', alpha=0.6)
plt.xlabel('recency')
plt.ylabel('clv_historical')
plt.title('Scatter Plot - recency vs clv_historical')

# Frequency vs clv_historical
plt.subplot(2, 3, 5)
plt.scatter(std_df['frequency'], std_df['clv_historical'],color='blue', alpha=0.6)
plt.xlabel('Frequency')
plt.ylabel('clv_historical')
plt.title('Scatter Plot - Frequency vs clv_historical')

# Frequency vs clv_historical
plt.subplot(2, 3, 6)
plt.scatter(std_df['monetary'], std_df['clv_historical'],color='red', alpha=0.6)
plt.xlabel('Monetary')
plt.ylabel('clv_historical')
plt.title('Scatter Plot - Monetary vs clv_historical')



plt.tight_layout()
plt.show()


# ## **Robust Normalization**

# In[49]:


scaler3 = pp.RobustScaler()
robust_rfm = scaler3.fit_transform(rfm_clv_n)
robust_rfm


# In[50]:


# Array → DataFrame

robust_df = pd.DataFrame(robust_rfm , columns = relevant_cols)
robust_df.head()


# In[51]:


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

# recency vs clv_historical
plt.subplot(2, 3, 4)
plt.scatter(robust_df['recency'], robust_df['clv_historical'],color='green', alpha=0.6)
plt.xlabel('recency')
plt.ylabel('clv_historical')
plt.title('Scatter Plot - recency vs clv_historical')

# Frequency vs clv_historical
plt.subplot(2, 3, 5)
plt.scatter(robust_df['frequency'], robust_df['clv_historical'], color='blue', alpha=0.6)
plt.xlabel('Frequency')
plt.ylabel('clv_historical')
plt.title('Scatter Plot - Frequency vs clv_historical')

# Frequency vs clv_historical
plt.subplot(2, 3, 6)
plt.scatter(robust_df['monetary'], robust_df['clv_historical'],color='red', alpha=0.6)
plt.xlabel('Monetary')
plt.ylabel('clv_historical')
plt.title('Scatter Plot - Monetary vs clv_historical')



plt.tight_layout()
plt.show()


# # **Clustering - KMeans - ElbowPlot**

# In[52]:


from sklearn.cluster import KMeans


# In[53]:


def find_best_clusters(df, maximum_K):

    clusters_centers = []
    k_values = []

    for k in range(1, maximum_K):

        kmeans_model = KMeans(n_clusters = k)
        kmeans_model.fit(df)

        clusters_centers.append(kmeans_model.inertia_)
        k_values.append(k)


    return clusters_centers, k_values


# In[54]:


def generate_elbow_plot(clusters_centers, k_values):

    figure = plt.subplots(figsize = (12, 6))
    plt.plot(k_values, clusters_centers, 'o-', color = 'orange')
    plt.xlabel("Number of Clusters (K)")
    plt.ylabel("Cluster Inertia")
    plt.title("Elbow Plot of KMeans")
    plt.show()


# In[55]:


clusters_centers, k_values = find_best_clusters(std_rfm, 18)

generate_elbow_plot(clusters_centers, k_values)


# In[56]:


kmeans_model = KMeans(n_clusters = 5)
kmeans_model.fit(std_rfm)


# In[57]:


std_df["clusters"] = kmeans_model.labels_
std_df.head()


# In[58]:


plt.figure(figsize=(18,8))

# Recency vs Frequency
plt.subplot(2, 3, 1)
plt.scatter(std_df['recency'], std_df['frequency'], c = std_df["clusters"], alpha=0.5)
plt.xlabel('Recency')
plt.ylabel('Frequency')
plt.title('Recency vs Frequency')

# Recency vs Monetary
plt.subplot(2, 3, 2)
plt.scatter(std_df['recency'], std_df['monetary'], c = std_df["clusters"], alpha=0.5)
plt.xlabel('Recency')
plt.ylabel('Monetary')
plt.title('Recency vs Monetary')

# Frequency vs Monetary
plt.subplot(2, 3, 3)
plt.scatter(std_df['frequency'], std_df['monetary'], c = std_df["clusters"], alpha=0.5)
plt.xlabel('Frequency')
plt.ylabel('Monetary')
plt.title('Frequency vs Monetary')

# recency vs clv_historical
plt.subplot(2, 3, 4)
plt.scatter(std_df['recency'], std_df['clv_historical'], c = std_df["clusters"] , alpha=0.6)
plt.xlabel('recency')
plt.ylabel('clv_historical')
plt.title('Scatter Plot - recency vs clv_historical')

# Frequency vs clv_historical
plt.subplot(2, 3, 5)
plt.scatter(std_df['frequency'], std_df['clv_historical'], c = std_df["clusters"], alpha=0.6)
plt.xlabel('Frequency')
plt.ylabel('clv_historical')
plt.title('Scatter Plot - Frequency vs clv_historical')

# Frequency vs clv_historical
plt.subplot(2, 3, 6)
plt.scatter(std_df['monetary'], std_df['clv_historical'], c = std_df["clusters"], alpha=0.6)
plt.xlabel('Monetary')
plt.ylabel('clv_historical')
plt.title('Scatter Plot - Monetary vs clv_historical')


plt.tight_layout()
plt.show()


# In[59]:


rfm_clv


# In[60]:


rfm_clv['clusters'] = kmeans_model.labels_
rfm_clv.head()


# In[61]:


std_df


# In[62]:


std_df['customer_id'] = rfm_clv['customer_id'].values
std_df


# In[63]:


std_df.to_csv('../exports/rfm_clv_kmeans_normalized_data.csv')


# In[64]:


std_df.to_excel('../exports/rfm_clv_kmeans_normalized_data.xlsx')


# In[65]:


fig, axes = plt.subplots(1, 2, figsize=(14,6))  # 1 ردیف، 2 ستون

sbn.stripplot(x='clusters', y='recency' , data = rfm_clv ,  ax=axes[0])
axes[0].set_title("Recency by Clusters - rfm_clv")


sbn.stripplot(x='clusters', y='recency' , data = std_df ,  ax=axes[1])
axes[1].set_title("Recency by Clusters - std_df")


plt.tight_layout()
plt.show()


# In[66]:


fig, axes = plt.subplots(1, 2, figsize=(14,6))  # 1 ردیف، 2 ستون

sbn.stripplot(x='clusters', y='frequency' , data = rfm_clv ,  ax=axes[0])
axes[0].set_title("Recency by Frequency - rfm_clv")


sbn.stripplot(x='clusters', y='frequency' , data = std_df ,  ax=axes[1])
axes[1].set_title("Recency by Frequency - std_df")


plt.tight_layout()
plt.show()


# In[67]:


fig, axes = plt.subplots(1, 2, figsize=(14,6))  # 1 ردیف، 2 ستون

sbn.stripplot(x='clusters', y='monetary' , data = rfm_clv ,  ax=axes[0])
axes[0].set_title("Recency by Monetary - rfm_clv")


sbn.stripplot(x='clusters', y='monetary' , data = std_df ,  ax=axes[1])
axes[1].set_title("Recency by Monetary - std_df")


plt.tight_layout()
plt.show()


# In[76]:


fig, axes = plt.subplots(1, 2, figsize=(14,6))  # 1 ردیف، 2 ستون

sbn.stripplot(x='clusters', y='clv_historical' , data = rfm_clv ,  ax=axes[0])
axes[0].set_title("Recency by CLV_Historical - rfm_clv")


sbn.stripplot(x='clusters', y='clv_historical' , data = std_df ,  ax=axes[1])
axes[1].set_title("Recency by CLV_Historical - std_df")


plt.tight_layout()
plt.show()


# In[68]:


cluster_profile = std_df.groupby("clusters").agg({
                                                'recency' : 'mean',
                                                'frequency' : 'mean',
                                                'monetary' : 'mean',
                                                'clv_historical' : 'mean',
                                                'customer_id' : 'count'
}).rename(columns = {"customer_id" : "customer_count"})

cluster_profile


# In[69]:


cluster_profile.to_csv('../exports/rfm_clv_kmeans_normalized_data_mean.csv')


# In[70]:


cluster_profile.to_excel('../exports/rfm_clv_kmeans_normalized_data_mean.xlsx')


# In[71]:


cluster_customers = std_df.groupby("clusters")['customer_id'].apply(list)
cluster_customers


# In[72]:


cluster_customers.columns = ['clusters', 'customer_id']
cluster_customers


# In[73]:


cluster_customers_exploded = std_df[['clusters', 'customer_id']].sort_values(by='clusters')
cluster_customers_exploded


# In[74]:


cluster_customers_exploded[cluster_customers_exploded['clusters']==3]


# In[ ]:




