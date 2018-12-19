
# coding: utf-8

# In[1]:


get_ipython().run_line_magic('matplotlib', 'notebook')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.style

#https://data.world/oecd/gender-wage-gap
# The gender wage gap is unadjusted and is defined as the difference 
# between median earnings of men and women relative to median earnings of men. 
# Data refer to full-time employees and to self-employed.


# In[2]:


df = pd.read_csv('gender_wage_gap.csv')
df.head()


# In[3]:


df = df[df['SUBJECT'] == 'TOT']


# In[4]:


df = df[['LOCATION', 'TIME', 'Value']]
df.columns = ['Country', 'Year', 'Gap %']

df.head()


# In[5]:


df.columns


# In[6]:


countries = ['AUS', 'JPN', 'SWE', 'GBR', 'USA']
df = df[df['Country'].isin(countries)]
df.head()


# In[7]:


df['Country'].unique()


# In[8]:


plt.style.available


# In[9]:


df.columns


# In[14]:


plt.style.use('seaborn-poster')
fig, ax = plt.subplots(figsize = (10, 7))

for key, grp  in df.groupby('Country'):
    ax = grp.plot(ax = ax, kind = 'line', x = 'Year', y = 'Gap %', label = key, legend = False)

ax.tick_params(axis = 'both', which = 'major', labelsize = 14)
ax.axhline(y = 0, color = 'black', linewidth = 1.3, alpha = 0.7)
ax.set_yticklabels(labels = ['-10 ', '0   ', '10  ', '20  ', '30  ', '40%'])
ax.set_xlim(left = 1970, right = 2017)
ax.xaxis.label.set_visible(False)
ax.grid(False)

ax.text(x = 1968, y = 55, s = 'Gender Pay Gap for 5 major countries (1970 - 2016)', fontsize = 18)
ax.text(x = 1968, y = 52, s = 'The gender wage gap is unadjusted and is defined as the difference between', fontsize = 14) 
ax.text(x = 1968, y = 50, s = 'median earnings of men and women relative to median earnings of men.', fontsize = 14)

plt.legend()
plt.show()

