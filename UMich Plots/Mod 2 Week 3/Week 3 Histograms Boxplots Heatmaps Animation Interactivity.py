
# coding: utf-8

# # SUBPLOTS

# In[1]:


get_ipython().run_line_magic('matplotlib', 'notebook')


# In[2]:


import matplotlib.pyplot as plt
import numpy as np

get_ipython().run_line_magic('pinfo', 'plt.subplot')


# In[3]:


plt.figure()
plt.subplot(1, 2, 1) #1 row, 2 columns, 1st axis current axis

lin_data = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9])

plt.plot(lin_data, '-o')


# In[4]:


exp_data = lin_data ** 2

plt.subplot(1, 2, 2)
plt.plot(exp_data, '-o')


# In[5]:


plt.subplot(1, 2, 1)
plt.plot(exp_data, '-x')


# In[6]:


plt.figure()

ax1 = plt.subplot(1, 2, 1)
plt.plot(lin_data, '-o')
ax2 = plt.subplot(1, 2, 2, sharey = ax1)
plt.plot(exp_data, '-x')


# In[7]:


plt.figure()
plt.subplot(1, 2, 1) == plt.subplot(121)


# In[8]:


fig, ((ax1, ax2, ax3), (ax4, ax5, ax6), (ax7, ax8, ax9)) = plt.subplots(3, 3, sharex = True, sharey = True)

ax5.plot(lin_data, '-')


# In[9]:


for ax in plt.gcf().get_axes():  
    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_visible(True)


# In[10]:


plt.gcf().canvas.draw()


# # Histograms

# In[11]:


fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, sharex = True)
axs = [ax1, ax2, ax3, ax4]

for n in range(0, len(axs)):
    sample_size = 10**(n+1)
    sample = np.random.normal(loc = 0.0, scale = 1.0, size = sample_size)
    axs[n].hist(sample)
    axs[n].set_title('n = {}'.format(sample_size))


# In[12]:


fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, sharex = True)
axs = [ax1, ax2, ax3, ax4]

for n in range(0, len(axs)):
    sample_size = 10**(n+1)
    sample = np.random.normal(loc = 0.0, scale = 1.0, size = sample_size)
    axs[n].hist(sample, bins = 100)
    axs[n].set_title('n = {}'.format(sample_size))


# In[13]:


plt.figure()
Y = np.random.normal(loc = 0.0, scale = 1.0, size = 10000)
X = np.random.random(size = 10000)
plt.scatter(X, Y)


# In[14]:


import matplotlib.gridspec as gridspec

plt.figure()
gspec = gridspec.GridSpec(3, 3)

top_histogram = plt.subplot(gspec[0,1:])
side_histogram = plt.subplot(gspec[1:, 0])
lower_right = plt.subplot(gspec[1:, 1:])


# In[15]:


Y = np.random.normal(loc = 0.0, scale = 1.0, size = 10000)
X = np.random.random(size = 10000)
lower_right.scatter(X, Y)
top_histogram.hist(X, bins = 100)
s = side_histogram.hist(Y, bins = 100, orientation = 'horizontal')


# In[16]:


top_histogram.clear()
top_histogram.hist(X, bins = 100, normed = True)
side_histogram.clear()
side_histogram.hist(Y, bins = 100, orientation = 'horizontal', normed = True)

side_histogram.invert_xaxis()


# In[17]:


for ax in [top_histogram, lower_right]:
    ax.set_xlim(0, 1)
for ax in [side_histogram, lower_right]:
    ax.set_ylim(-5, 5)


# # BOXPLOTS

# In[5]:


import pandas as pd


# In[19]:


norm_sample = np.random.normal(loc = 0.0, scale = 1.0, size = 10000)
rand_sample = np.random.random(size = 10000)
gamma_sample = np.random.gamma(2, size = 10000)


# In[20]:


df = pd.DataFrame({'normal': norm_sample,
                   'random': rand_sample,
                   'gamma': gamma_sample})


# In[21]:


df.describe()


# In[22]:


plt.figure()
_ = plt.boxplot(df['normal'], whis = 'range') # _ can be used as a variable name, if you dont want to use it later


# In[23]:


plt.clf()
_ = plt.boxplot([df['normal'], df['gamma'], df['random']], whis = 'range')


# In[24]:


plt.figure()
_ = plt.hist(df['gamma'], bins = 100)


# In[25]:


import mpl_toolkits.axes_grid1.inset_locator as mpl_il


# In[26]:


plt.figure()
plt.boxplot([df['normal'], df['random'], df['gamma']], whis = 'range')
ax2 = mpl_il.inset_axes(plt.gca(), width = '60%', height = '40%', loc = 2)
ax2.hist(df['gamma'], bins = 100)
ax2.margins(x = 0.5)


# In[27]:


ax2.yaxis.tick_right()


# In[28]:


plt.figure()
_ = plt.boxplot([df['normal'], df['random'], df['gamma']])


# # Heatmaps

# In[29]:


plt.figure()

Y = np.random.normal(loc = 0.0, scale = 1.0, size = 10000) 
X = np.random.random(size = 10000)

_ = plt.hist2d(X, Y, bins = 25)


# In[30]:


plt.figure()
_ = plt.hist2d(X, Y, bins = 100)


# In[31]:


plt.colorbar()


# # Animation

# In[32]:


import matplotlib.animation as animation


# In[35]:


n = 100
x = np.random.randn(n)

def update(curr):
    if curr == n:
        a.event_source.stop()
    plt.cla()
    bins = np.arange(-4, 4, 0.5)
    plt.hist(x[:curr], bins = bins)
    plt.axis([-4, 4, 0, 30])
    plt.gca().set_title('Sampling the Normal Distribution')
    plt.gca().set_ylabel('Frequency')
    plt.gca().set_xlabel('Value')
    plt.annotate('n = {}'.format(curr), [3, 27])


# In[36]:


fig = plt.figure()
a = animation.FuncAnimation(fig, update, interval = 100)


# # Interactivity

# In[10]:


plt.figure()
data = np.random.randn(10)
plt.plot(data)

def onclick(event):
    plt.cla()
    plt.plot(data)
    plt.gca().set_title('Event at pixels{}, {} {} and data {}, {}'.format(event.x,
                                                                       event.y,
                                                                       '\n',
                                                                       event.xdata,
                                                                       event.ydata))

plt.gcf().canvas.mpl_connect('button_press_event', onclick)


# In[6]:


from random import shuffle

origins = ['India', 'Brazil', 'USA', 'UK', 'China', 'Mexico', 'Chile', 'Iraq', 'Canada', 'Germany']

shuffle(origins)

df = pd.DataFrame({'Height': np.random.rand(10),
                   'Weight': np.random.rand(10),
                   'Origin': origins})


# In[7]:


df


# In[8]:


plt.figure()
plt.scatter(df['Height'], df['Weight'], picker = 5)
plt.gca().set_xlabel('Height')
plt.gca().set_ylabel('Weight')


# In[9]:


def onpick(event):
    origin = df.iloc[event.ind[0]]['Origin']
    plt.gca().set_title('Selected item came from {}'.format(origin))

plt.gcf().canvas.mpl_connect('pick_event', onpick)

