
# coding: utf-8

# In[1]:


get_ipython().run_line_magic('matplotlib', 'notebook')

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.animation as animation

# pd.options.display.float_format = '{:,.2f}'.format

np.random.seed(12345)

df = pd.DataFrame([np.random.normal(32000,200000,3650), 
                   np.random.normal(43000,100000,3650), 
                   np.random.normal(43500,140000,3650), 
                   np.random.normal(48000,70000,3650)], 
                  index=[1992,1993,1994,1995])

df.index.name = 'Year'
df['Sum'] = (df.select_dtypes(float).sum(1))/1e8

df


# In[14]:


fig, ax = plt.subplots()

pos = np.arange(len(df.index))

bars = plt.bar(pos, df['Sum'], align = 'center', linewidth = 0)


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
    
    plt.gca().set_title('Clicked at {}, {} and data is {}, {}'.format(event.x, event.y, '\n', event.xdata, event.ydata))
    
    plt.axhline(y = ev_y, xmin = 0, xmax = 1, ls = '--', c = 'b')
    
    bars = plt.bar(pos, df['Sum'], align = 'center', linewidth = 0)
    
    plt.xticks(pos, df.index, alpha = 0.8)

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


# In[ ]:


def color_bars(marker):
    for bar in bars:
        if (marker < bar.get_height()):
            bar.set_color('r')
        elif (marker > bar.get_height()):
            bar.set_color('b')
        elif ((1.47 > bar.get_height()) or (1.36 < bar.get_height())):
            bar.set_color('w')

color_bars(1.43)


# In[ ]:


import matplotlib.animation as animation


# In[ ]:


plt.bar.color = 'r'


# In[ ]:


marker*1e7


# In[ ]:


1.42000000.0
1.2158919228573697


# In[ ]:


for bar in bars:
    marker - 
    print('{:.2f}'.format bar.get_height())


# In[ ]:


import numpy as np
import matplotlib.pyplot as plt

t = np.arange(-1, 2, .01)
s = np.sin(2 * np.pi * t)

plt.figure()

plt.plot(t, s)
# Draw a thick red hline at y=0 that spans the xrange
plt.axhline(linewidth=8, color='#d62728')

# Draw a default hline at y=1 that spans the xrange
plt.axhline(y=1)

# Draw a default vline at x=1 that spans the yrange
plt.axvline(x=1)

# Draw a thick blue vline at x=0 that spans the upper quadrant of the yrange
plt.axvline(x=0, ymin=0.75, linewidth=8, color='#1f77b4')

# Draw a default hline at y=.5 that spans the middle half of the axes
plt.axhline(y=.5, xmin=0.25, xmax=0.75)



plt.show()


# In[ ]:


df2 = df.T

df3 = pd.DataFrame(df2.apply(lambda x: '{:.2f}'.format(float(x.sum()))))

df3


# In[ ]:


import matplotlib.pyplot as plt

plt.figure()

pos = np.arange(len(df3.index))

plt.bar(pos, df3[0], align = 'center', linewidth = 0)

plt.xticks(pos, df3.index, alpha = 0.8)


# In[ ]:


df['Sum'] = df.apply(lambda x: '{:.2f}'.format(float(x.sum())))


# In[ ]:


pos = np.arange(len(df3.index))


# In[ ]:


pos


# In[ ]:


languages =['Python', 'SQL', 'Java', 'C++', 'JavaScript']
pos = np.arange(len(languages))
popularity = [56, 39, 34, 34, 29]


# In[ ]:


pos


# In[ ]:


import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
mpl.get_backend()

plt.figure()

languages =['Python', 'SQL', 'Java', 'C++', 'JavaScript']
pos = np.arange(len(languages))
popularity = [56, 39, 34, 34, 29]

# change the bar colors to be less bright blue
bars = plt.bar(pos, popularity, align='center', linewidth=0, color='lightslategrey')
# make one bar, the python bar, a contrasting color
bars[0].set_color('#1F77B4')

# soften all labels by turning grey
plt.xticks(pos, languages, alpha=0.8)
# plt.ylabel('% Popularity', alpha=0.8)
plt.title('Top 5 Languages for Math & Data \nby % popularity on Stack Overflow', alpha=0.8)

# remove the frame of the chart
for spine in plt.gca().spines.values():
    spine.set_visible(False)
    
# remove all the ticks (both axes), and tick labels on the Y axis
plt.tick_params(top='off', bottom='off', left='off', right='off', labelleft='off', labelbottom='on')

# direct label each bar with Y axis values
for bar in bars:
    plt.gca().text(bar.get_x() + bar.get_width()/2, bar.get_height() - 5, str(int(bar.get_height())) + '%', 
                 ha='center', color='w', fontsize=11)
plt.show()

