
# coding: utf-8

# In[42]:


get_ipython().run_line_magic('matplotlib', 'notebook')

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.animation as animation
from scipy import stats

# pd.options.display.float_format = '{:,.2f}'.format

np.random.seed(12345)

df = pd.DataFrame(np.c_[np.random.normal(32000,200000,3650), 
                   np.random.normal(43000,100000,3650), 
                   np.random.normal(43500,140000,3650), 
                   np.random.normal(48000,70000,3650)], 
                  columns=[1992,1993,1994,1995])

# df.index.name = 'Year'
# df['Sum'] = (df.select_dtypes(float).sum(1))/1e8

# std = df.std()

df


# In[31]:


fig, ax = plt.subplots()

pos = np.arange(len(df.index))

bars = plt.bar(pos, df['Sum'], yerr=std, align = 'center', linewidth = 0)

def color_bars(marker):
    for bar in bars:
        if ((marker + 0.05) < bar.get_height()):
            bar.set_color('r')
        elif ((marker - 0.05) > bar.get_height()):
            bar.set_color('b')
        else:
            bar.set_color('grey')

def onclick(event):
    
    plt.cla()
    
    bars = plt.bar(pos, df['Sum'], align = 'center', linewidth = 0)
    
    ev_y = event.ydata
    
    plt.gca().set_title('Interactive plot with Y-Axis '.format(event.x, event.y, '\n', event.xdata, event.ydata))
    
    plt.axhline(y = ev_y, xmin = 0, xmax = 1, ls = '--', c = 'b')
    
#     bars = plt.bar(pos, df['Sum'], align = 'center', linewidth = 0)
    
    plt.xticks(pos, df.index, alpha = 0.8)
    
    plt.gca().set_ylabel('')

    plt.bar.color = 'r'
    
    marker = ev_y

#     color_bars(marker)
    
    for bar in bars:
        if ((marker + 0.05) < bar.get_height()):
            bar.set_color('r')
        elif ((marker - 0.05) > bar.get_height()):
            bar.set_color('b')
        else:
            bar.set_color('grey')
    
    plt.show()
    
plt.gcf().canvas.mpl_connect('button_press_event', onclick)

# plt.axhline(y = marker, xmin = 0, xmax = 1, ls = '--', c = 'y')


# In[25]:


df.shape


# In[38]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# plt.figure()

np.random.seed(12345)
df = pd.DataFrame(np.c_[np.random.normal(-10,200,100), 
                   np.random.normal(42,150,100), 
                   np.random.normal(0,120,100), 
                   np.random.normal(-5,57,100)], 
                  columns=[2012,2013,2014,2015])

df

