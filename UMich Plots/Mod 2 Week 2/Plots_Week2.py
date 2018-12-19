
# coding: utf-8

# In[1]:


get_ipython().run_line_magic('matplotlib', 'notebook')


# In[2]:


import matplotlib as mpl
mpl.get_backend()


# In[3]:


import matplotlib.pyplot as plt
get_ipython().run_line_magic('pinfo', 'plt.plot')


# In[4]:


plt.plot(2, 3, '.')


# In[5]:


plt.plot(4, 5, '.')


# In[6]:


from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure

fig = Figure()
canvas = FigureCanvasAgg(fig)

ax = fig.add_subplot(111)
ax.plot(3, 2, '.')

canvas.print_png('test.png')


# In[7]:


get_ipython().run_cell_magic('html', '', "<img src = 'test.png' />")


# In[8]:


plt.figure()
plt.plot(3, 2, 'o')
ax = plt.gca()
ax.axis([0, 6, 0, 10])


# In[9]:


plt.figure()
plt.plot(3, 2, 'o')
plt.plot(4, 6, 'o')
plt.plot(2.5, 2.5, 'o')


# In[10]:


ax = plt.gca()
ax.get_children()


# # Scatterplot

# In[11]:


import numpy as np

x = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9])
y = x

plt.figure()
plt.scatter(x, y)


# In[12]:


x = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9])
y = x
colors = ['blue']*(len(x) - 1)
colors.append('yellow')

plt.figure()
plt.scatter(x, y, s = 100, c = colors)


# In[13]:


zip_gen = zip([1, 2, 3, 4, 5], [6, 7, 8, 9, 10])
list(zip_gen)


# In[14]:


zip_gen1 = zip([1, 2, 3, 4, 5], [6, 7, 8, 9, 10])
x, y = zip(*zip_gen1)
print(x)
print(y)


# In[15]:


plt.figure()
plt.scatter(x[:2], y[:2], s = 100, c = 'red', label = 'Tall Students')
plt.scatter(x[2:], y[2:], s = 100, c = 'blue', label = 'Short Students')


# In[16]:


plt.xlabel('No. of times a student kicked a ball')
plt.ylabel('Grade of the student')
plt.title('Relationship between kicking a ball and grades')


# In[17]:


plt.legend()


# In[18]:


plt.legend(loc = 4, frameon = False, title = 'Legend')


# In[19]:


plt.gca().get_children()


# In[20]:


legend = plt.gca().get_children()[-2]
legend


# In[21]:


legend.get_children()[0].get_children()[1].get_children()[0].get_children()


# In[22]:


from matplotlib.artist import Artist

def get_gc(art, depth = 0):
    if isinstance(art, Artist):
        print(' ' * depth + str(art))
        for child in art.get_children():
            get_gc(child, depth + 2)
            
get_gc(legend)


# # Line Plots

# In[23]:


import numpy as np

lin_data = np.array([1, 2, 3, 4, 5, 6, 7, 8])
quad_data = lin_data **2

plt.figure()
plt.plot(lin_data, '-o', quad_data, '-o')


# In[24]:


plt.plot([22, 44, 55], '--r')


# In[25]:


plt.xlabel('Some Data')
plt.ylabel('Some Other Data')
plt.title('A title')
plt.legend(['Baseline', 'Competition', 'Us'])


# In[26]:


plt.gca().fill_between(range(len(lin_data)),
                       lin_data, quad_data, facecolor = 'blue',
                       alpha = 0.25)


# In[27]:


#Doesnt work coz datetime isn't passed to matplotlib in the right format
plt.figure()

obser_dates = np.arange('2017-01-01', '2017-01-09', dtype = 'datetime64[D]')
plt.plot(obser_dates, lin_data, '-o',
         obser_dates, quad_data, '-o')


# In[28]:


import pandas as pd


# In[29]:


plt.figure()
obser_dates = np.arange('2017-01-01', '2017-01-09', dtype = 'datetime64[D]')
obser_dates = list(map(pd.to_datetime, obser_dates))

plt.plot(obser_dates, lin_data, '-o',
         obser_dates, quad_data, '-o')


# In[30]:


x = plt.gca().xaxis

for item in x.get_ticklabels():
    item.set_rotation(45)


# In[31]:


plt.subplots_adjust(bottom = 0.25)


# In[32]:


ax = plt.gca()
ax.set_xlabel('Date')
ax.set_ylabel('Units')
ax.set_title('Quad Vs. Linear perfomance')


# In[33]:


ax.set_title('Quadratic ($x^2$) Vs. Linear($x$) performance')


# # Bar Charts

# In[34]:


plt.figure()
xvals = range(len(lin_data))
plt.bar(xvals, lin_data, width = 0.4)


# In[35]:


new_xvals = []
for item in xvals:
    new_xvals.append(item+0.4)
    
plt.bar(new_xvals, quad_data, width = 0.3, color = 'red')


# In[36]:


from random import randint
lin_err = [randint(0, 15) for x in range(len(lin_data))]
plt.bar(xvals, lin_data, width = 0.3, yerr = lin_err)


# In[37]:


plt.figure()
xvals = range(len(lin_data))
plt.bar(xvals, lin_data, width = 0.3, color = 'blue')
plt.bar(xvals, quad_data, width = 0.5, bottom = lin_data, color = 'r')


# In[38]:


plt.figure()
xvals = range(len(lin_data))
plt.barh(xvals, lin_data, height = 0.3, color = 'b')
plt.barh(xvals, quad_data, height = 0.3, left = lin_data, color = 'r')


# In[39]:


import matplotlib.pyplot as plt
import numpy as np

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

# remove all the ticks (both axes), and tick labels on the Y axis
plt.tick_params(top='off', bottom='off', left='off', right='off', labelleft='off', labelbottom='on')

# remove the frame of the chart
for spine in plt.gca().spines.values():
    spine.set_visible(False)

# direct label each bar with Y axis values
for bar in bars:
    plt.gca().text(bar.get_x() + bar.get_width()/2, bar.get_height() - 5, str(int(bar.get_height())) + '%', 
                 ha='center', color='w', fontsize=11)

plt.show()

