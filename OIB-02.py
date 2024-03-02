#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns
import os


# In[2]:


os.getcwd()


# In[5]:


os.chdir('C:\\Users\\Abhi\\documents\\readings')


# In[6]:


df=pd.read_csv('ifood_df.csv')


# In[7]:


df.head()


# In[6]:


import warnings
warnings.filterwarnings("ignore")
from scipy.stats import pointbiserialr


# In[7]:


df.describe()


# In[8]:


df.shape


# In[9]:


df.isnull().sum()


# In[11]:


df.nunique()


# In[13]:


df.drop(columns=['Z_CostContact','Z_Revenue'],inplace=True)


# In[14]:


df.shape


# In[16]:


plt.figure(figsize=(6,4))
sns.boxplot(data=df,y='MntTotal')
plt.ylabel='MntTotal'
plt.title('Boxplot of MntTotal')
plt.show()


# In[18]:


q1=df['MntTotal'].quantile(0.25)
q3=df['MntTotal'].quantile(0.75)
IQR=q3-q1
lower_bound=q1-1.5*IQR
upper_bound=q3+1.5*IQR
outliers=df[(df['MntTotal']<lower_bound)|(df['MntTotal']>upper_bound)]
outliers.head()


# In[22]:


plt.figure(figsize=(6,4))
sns.boxplot(data=df,y='Income',palette='viridis')
plt.ylabel=('Income')
plt.title=('Boxplot for Income')
plt.show()


# In[26]:


plt.figure(figsize=(6,5))
sns.histplot(data=df,x='Income',bins=30,kde=True)
plt.xlabel='Income'
plt.ylabel='frequency'
plt.title=('Histrogram for Income')
plt.show()


# In[27]:


plt.figure(figsize=(6,5))
sns.histplot(data=df,x='Age',bins=30,kde=True)
plt.xlabel='Age'
plt.ylabel='frequency'
plt.title=('Histrogram for Age')
plt.show()


# In[29]:


print("Skewness: %f" % df['Age'].skew())
print("Kurtosis: %f" % df['Age'].kurt())


# In[30]:


cols_demographics = ['Income','Age']
cols_children = ['Kidhome', 'Teenhome']
cols_marital = ['marital_Divorced', 'marital_Married','marital_Single', 'marital_Together', 'marital_Widow']
cols_mnt = ['MntTotal', 'MntRegularProds','MntWines', 'MntFruits', 'MntMeatProducts', 'MntFishProducts', 'MntSweetProducts', 'MntGoldProds']
cols_communication = ['Complain', 'Response', 'Customer_Days']
cols_campaigns = ['AcceptedCmpOverall', 'AcceptedCmp1', 'AcceptedCmp2', 'AcceptedCmp3', 'AcceptedCmp4', 'AcceptedCmp5']
cols_source_of_purchase = ['NumDealsPurchases', 'NumWebPurchases','NumCatalogPurchases', 'NumStorePurchases', 'NumWebVisitsMonth']
cols_education = ['education_2n Cycle', 'education_Basic', 'education_Graduation', 'education_Master', 'education_PhD']


# In[32]:


corr_matrix = df[['MntTotal']+cols_demographics+cols_children].corr()
plt.figure(figsize=(6,6))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', linewidths=0.5)
plt.title('Correlation Matrix Heatmap')
plt.show()


# # Feature Engeneering 

# In[38]:


def get_marital_status(row):
    if row['marital_Divorced'] == 1:
        return 'Divorced'
    elif row['marital_Married'] == 1:
        return 'Married'
    elif row['marital_Single'] == 1:
        return 'Single'
    elif row['marital_Together'] == 1:
        return 'Together'
    elif row['marital_Widow'] == 1:
        return 'Widow'
    else:
        return 'Unknown'
df['Marital'] = df.apply(get_marital_status,axis=1)


# In[40]:


plt.figure(figsize=(6,5))
sns.barplot(data=df,x='Marital',y='MntTotal',palette='viridis')
plt.xlabel=('Marital Status')
plt.ylabel=('MntTotal')
plt.title=('MntTotal by marital status')
plt.show()


# In[13]:


# feature engennering 'In_relationship'
def get_relationship(row):
    if row['marital_Married'] ==1:
        return 1
    elif row['marital_Together'] == 1:
        return 1
    else:
        return 0
df['In_relationship'] = df.apply(get_relationship, axis=1)
df.head()    


# In[11]:


from sklearn.cluster import KMeans


# In[15]:


from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
cols_for_clustering = ['Income', 'MntTotal', 'In_relationship']
df_scaled = df.copy()
df_scaled[cols_for_clustering] = scaler.fit_transform(df[cols_for_clustering])
df_scaled[cols_for_clustering].describe()


# In[18]:


from sklearn import decomposition
pca = decomposition.PCA(n_components = 2)
pca_res = pca.fit_transform(df_scaled[cols_for_clustering])
df_scaled['pc1'] = pca_res[:,0]
df_scaled['pc2'] = pca_res[:,1]


# In[ ]:





# In[19]:


X = df_scaled[cols_for_clustering]
inertia_list = []
for K in range(2,10):
    inertia = KMeans(n_clusters=K, random_state=7).fit(X).inertia_
    inertia_list.append(inertia)


# In[21]:


plt.figure(figsize=[5,4])
plt.plot(range(2,10), inertia_list, color=(54 / 255, 113 / 255, 130 / 255))
plt.title("Inertia vs. Number of Clusters")
plt.xlabel("Number of Clusters (K)")
plt.ylabel("Inertia")
plt.show()


# In[ ]:





# In[ ]:





# In[ ]:




