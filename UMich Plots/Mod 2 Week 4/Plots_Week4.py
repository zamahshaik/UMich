
# coding: utf-8

# In[9]:


get_ipython().run_line_magic('matplotlib', 'notebook')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# In[2]:


plt.style.available


# In[3]:


plt.style.use('seaborn-colorblind')


# # DataFrame

# In[4]:


np.random.seed(123)

df = pd.DataFrame({'A': np.random.randn(365).cumsum(0),
                   'B': np.random.randn(365).cumsum(0) +20,
                   'C': np.random.randn(365).cumsum(0) -20},
                   index = pd.date_range('1/1/2017', periods = 365))

df.head()


# In[6]:


df.plot(); # ; supresses unwanted output


# In[7]:


df.plot('A', 'B', kind = 'scatter')


# In[10]:


df.plot.scatter('A', 'C', c = 'B', s = df['B'], colormap = 'viridis')


# In[11]:


ax = df.plot.scatter('A', 'C', c = 'B', s = df['B'], colormap = 'viridis')
ax.set_aspect('equal')


# In[12]:


df.plot.box();


# In[13]:


df.plot.hist(alpha = 0.7)


# In[14]:


df.plot.kde();


# # pandas.tools.plotting

# In[15]:


iris = pd.read_csv('iris.csv')
iris.head()


# In[16]:


pd.tools.plotting.scatter_matrix(iris);


# In[18]:


plt.figure()
pd.tools.plotting.parallel_coordinates(iris, 'Name');


# In[1]:


get_ipython().run_line_magic('matplotlib', 'notebook')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# In[2]:


np.random.seed(1234)

v1 = pd.Series(np.random.normal(0, 10, 1000), name = 'v1') #1000 numbers normal distri mean of 0 SD = 10
v2 = pd.Series(2*v1 + np.random.normal(60, 15, 1000), name = 'v2')


# In[4]:


plt.figure()
plt.hist(v1, alpha = 0.7, bins = np.arange(-50, 150, 5), label = 'v1');
plt.hist(v2, alpha = 0.7, bins = np.arange(-50, 150, 5), label = 'v2');
plt.legend();


# In[5]:


plt.figure()
plt.hist([v1, v2], histtype = 'barstacked', normed = True);
v3 = np.concatenate((v1, v2))
sns.kdeplot(v3);


# In[6]:


plt.figure()
sns.distplot(v3, hist_kws={'color': 'teal'}, kde_kws = {'color': 'navy'});


# In[7]:


sns.jointplot(v1, v2, alpha = 0.4)


# In[9]:


grid = sns.jointplot(v1, v2, alpha = 0.4);
grid.ax_joint.set_aspect('equal');


# In[10]:


sns.jointplot(v1, v2, kind = 'hex')


# In[11]:


sns.set_style('white')
sns.jointplot(v1, v2, kind = 'kde', space = 0)


# In[12]:


iris = pd.read_csv('iris.csv')
iris.head()


# In[13]:


sns.pairplot(iris, hue = 'Name', diag_kind = 'kde');


# In[14]:


plt.figure(figsize = (10, 6))
plt.subplot(121)
sns.swarmplot('Name', 'PetalLength', data = iris);
plt.subplot(122)
sns.violinplot('Name', 'PetalLength', data = iris);

