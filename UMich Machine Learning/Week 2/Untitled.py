
# coding: utf-8

# In[1]:


import pandas as d1
import numpy as np


# In[9]:


df = pd.DataFrame([['Apple', 'Apple', 'Apple', 'Orange', 'Banana', 'Banana', 'Orange'],
                  ['Orange', 'Orange', 'Orange', 'Orange', 'Orange', 'Banana', 'Apple']])

df


# In[10]:


df = df.T


# In[11]:


df.columns = ['Col1', 'Col2']


# In[12]:


df


# In[19]:


fna = df['Col1'].unique()


# In[26]:


df['Col1'].value_counts()


# In[30]:


final = df.apply(pd.value_counts)


# In[31]:


final.T


# In[32]:


df = pd.read_csv('Data_Science_Topics_Survey.csv')


# In[33]:


df


# In[ ]:


df.columns = ['Data_Visualization', 'Machine_Learning', 'Data_Analysis_Statistics', '']

