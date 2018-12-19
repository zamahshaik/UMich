
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


# In[12]:


plt.style.use('fivethirtyeight')
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


# In[ ]:


"""
========================
Visualizing named colors
========================

Simple plot example with the named colors and its visual representation.
"""
from __future__ import division

import matplotlib.pyplot as plt
from matplotlib import colors as mcolors


colors = dict(mcolors.BASE_COLORS, **mcolors.CSS4_COLORS)

# Sort colors by hue, saturation, value and name.
by_hsv = sorted((tuple(mcolors.rgb_to_hsv(mcolors.to_rgba(color)[:3])), name)
                for name, color in colors.items())
sorted_names = [name for hsv, name in by_hsv]

n = len(sorted_names)
ncols = 4
nrows = n // ncols + 1

fig, ax = plt.subplots(figsize=(8, 5))

# Get height and width
X, Y = fig.get_dpi() * fig.get_size_inches()
h = Y / (nrows + 1)
w = X / ncols

for i, name in enumerate(sorted_names):
    col = i % ncols
    row = i // ncols
    y = Y - (row * h) - h

    xi_line = w * (col + 0.05)
    xf_line = w * (col + 0.25)
    xi_text = w * (col + 0.3)

    ax.text(xi_text, y, name, fontsize=(h * 0.8),
            horizontalalignment='left',
            verticalalignment='center')

    ax.hlines(y + h * 0.1, xi_line, xf_line,
              color=colors[name], linewidth=(h * 0.6))

ax.set_xlim(0, X)
ax.set_ylim(0, Y)
ax.set_axis_off()

fig.subplots_adjust(left=0, right=1,
                    top=1, bottom=0,
                    hspace=0, wspace=0)
plt.show()


# In[ ]:


fig, ax = plt.subplots(figsize = (10, 6))

for key, grp  in df.groupby('Country'):
    ax = grp.plot(ax = ax, kind = 'line', x = 'Year', y = 'Gap %', label = key, linestyle = 'solid', linewidth = 3)
plt.legend()
plt.show()


# In[ ]:


from pandas import *
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import datetime as dt

#The following part is just for generating something similar to your dataframe
date1 = "20140605"
date2 = "20140606"

d = {'date': Series([date1]*5 + [date2]*5),'template': Series(range(5)*2),'score': Series([random() for i in range(10)])} 

data = DataFrame(d)
#end of dataset generation

fig, ax = plt.subplots()

for temp in range(5):
    dat = data[data['template']==temp]
    dates =  dat['date']
    dates_f = [dt.datetime.strptime(date,'%Y%m%d') for date in dates]
    ax.plot(dates_f, dat['score'], label = "Template: {0}".format(temp))

plt.xlabel("Date")
plt.ylabel("Score")
ax.legend()
plt.show()


# In[ ]:


type(df)


# In[ ]:


df


# In[ ]:


df.head()

