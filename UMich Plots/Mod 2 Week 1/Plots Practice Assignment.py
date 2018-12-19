
# coding: utf-8

# In[3]:


import matplotlib.pyplot as plt
import numpy as np

get_ipython().run_line_magic('matplotlib', 'notebook')

# generate 4 random variables from the random, gamma, exponential, and uniform distributions
x1 = np.random.normal(-2.5, 1, 10000)
x2 = np.random.gamma(2, 1.5, 10000)
x3 = np.random.exponential(2, 10000)+7
x4 = np.random.uniform(14,20, 10000)

# plot the histograms
plt.figure(figsize=(9,3))
plt.hist(x1, normed=True, bins=20, alpha=0.5)
plt.hist(x2, normed=True, bins=20, alpha=0.5)
plt.hist(x3, normed=True, bins=20, alpha=0.5)
plt.hist(x4, normed=True, bins=20, alpha=0.5);
plt.axis([-7,21,0,0.6])

plt.text(x1.mean()-1.5, 0.5, 'x1\nNormal')
plt.text(x2.mean()-1.5, 0.5, 'x2\nGamma')
plt.text(x3.mean()-1.5, 0.5, 'x3\nExponential')
plt.text(x4.mean()-1.5, 0.5, 'x4\nUniform')


# In[4]:


''' Initial Bins start at 50.
    4 Subplots with for Normal, Gamma, Exponential and Uniform Distributions.
    Slider changes the number of bins.
    Reset button sets the bins to 100.
    Plots change as per the bins.
    
    Improvements needed: Plot titles disappear when the slider is used, tried correcting but couldn't. Help appreciated.
'''

import matplotlib.gridspec as gridspec
import matplotlib.animation as animation
from matplotlib.widgets import Slider, Button

fig = plt.figure()
gspec = gridspec.GridSpec(2, 2)
bins = 50

top_left = plt.subplot(gspec[0, 0])
plt.title('Normal Distribution')

top_right = plt.subplot(gspec[0, 1])
plt.title('Gamma Distribution')

lower_left = plt.subplot(gspec[1, 0])
plt.title('Exponential Distribution')

lower_right = plt.subplot(gspec[1, 1])
plt.title('Uniform Distribution')

top_left.hist(x1, bins = bins, color = 'red')
top_right.hist(x2, bins = bins, color = 'lightgreen')
lower_left.hist(x3, bins = bins, color = 'blue')
lower_right.hist(x4, bins = bins, color = 'yellow')

axslid = plt.axes([0.175, 0.4, 0.2, 0.03])#0.25, 0.1, 0.65, 0.03])
slid = Slider(axslid, 'Bins', 90, 300, valinit = 100, valstep = 5)

resetbn = plt.axes([0.19, 0.35, 0.1, 0.04])
button = Button(resetbn, 'Reset', hovercolor = '0.9')

def update(val):
    bin = int(slid.val)
    top_left.clear()
    top_right.clear()
    lower_left.clear()
    lower_right.clear()
    
    resetbn = plt.axes([0.19, 0.35, 0.1, 0.04])
    button = Button(resetbn, 'Reset', hovercolor = '0.9')

    top_left.hist(x1, bins = bin, color = 'blue')
    top_right.hist(x2, bins = bin, color = 'yellow')    
    lower_left.hist(x3, bins = bin, color = 'red')    
    lower_right.hist(x4, bins = bin, color = 'lightgreen')
    
    fig.canvas.draw_idle()
slid.on_changed(update)

def reset(event):
    plt.cla()
    slid.reset()
    resetbn = plt.axes([0.19, 0.35, 0.1, 0.04])
    button = Button(resetbn, 'Reset', hovercolor = '0.9')
button.on_clicked(reset)

plt.tight_layout()
plt.show()

