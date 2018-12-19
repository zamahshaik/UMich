
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer

cancer = load_breast_cancer()

print(cancer.DESCR)


# In[2]:


cancer.keys()


# In[3]:


type(cancer)


# In[4]:


len(cancer['feature_names'])


# In[5]:


df1 = pd.DataFrame(cancer.data, columns = cancer.feature_names)
df1['target'] = pd.Series(cancer.target)


# In[6]:


df1.shape


# In[7]:


columns = df1.columns.values.tolist()


# In[8]:


columns


# In[9]:


df1.index


# In[10]:


df1.target.unique()


# In[11]:


dft = pd.Series(df1['target'].value_counts()).rename({0: 'malignant', 1: 'benign'})


# In[12]:


dft


# In[13]:


type(dft)


# In[14]:


X = df1[df1.columns[0:30]]
y = df1['target']


# In[15]:


X


# In[16]:


y


# In[17]:


X.shape


# In[18]:


y.shape


# In[19]:


from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state = 0)


# In[20]:


X_train.shape


# In[21]:


X_test.shape


# In[22]:


y_train.shape


# In[23]:


y_test.shape


# In[24]:


from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier(n_neighbors = 1)
knn.fit(X_train, y_train)


# In[25]:


means = df1.mean()[:-1].values.reshape(1, -1)


# In[26]:


means


# In[27]:


cancer_predict = knn.predict(means)
type(cancer_predict)
print(cancer_predict)


# In[28]:


knn.predict(X_test)


# In[29]:


knn.score(X_test, y_test)

