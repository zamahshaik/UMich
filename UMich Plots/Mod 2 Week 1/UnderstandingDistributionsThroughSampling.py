
# coding: utf-8

# # Practice Assignment: Understanding Distributions Through Sampling
# 
# ** *This assignment is optional, and I encourage you to share your solutions with me and your peers in the discussion forums!* **
# 
# 
# To complete this assignment, create a code cell that:
# * Creates a number of subplots using the `pyplot subplots` or `matplotlib gridspec` functionality.
# * Creates an animation, pulling between 100 and 1000 samples from each of the random variables (`x1`, `x2`, `x3`, `x4`) for each plot and plotting this as we did in the lecture on animation.
# * **Bonus:** Go above and beyond and "wow" your classmates (and me!) by looking into matplotlib widgets and adding a widget which allows for parameterization of the distributions behind the sampling animations.
# 
# 
# Tips:
# * Before you start, think about the different ways you can create this visualization to be as interesting and effective as possible.
# * Take a look at the histograms below to get an idea of what the random variables look like, as well as their positioning with respect to one another. This is just a guide, so be creative in how you lay things out!
# * Try to keep the length of your animation reasonable (roughly between 10 and 30 seconds).

# In[1]:

import matplotlib.pyplot as plt
import numpy as np

get_ipython().magic('matplotlib notebook')

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


# In[6]:

from matplotlib.widgets import RadioButtons
import matplotlib.animation as animation
import numpy as np
import matplotlib.pyplot as plt

# create 2x2 grid of axis subplots
fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, sharex=False, sharey = True)
axs = [ax1,ax2,ax3,ax4]
colors= ['red','blue','green','yellow']
title = ['Normal','Gamma','Exponential','Uniform']
anno_x = [-1, 6.5, 13.5, 18.5]


x1 = np.random.normal(-2.5, 1, 10000)
x2 = np.random.gamma(2, 1.5, 10000)
x3 = np.random.exponential(2, 10000)+7
x4 = np.random.uniform(14,20, 10000)
#len(x1)




n = 1000
bns = 5

for x in range(0,len(axs)):
    axs[x].hist(eval('x'+str(x+1))[:100], bins=bns, alpha=0.5, color=colors[x], normed=True)
    axs[x].set_title('{}'.format(title[x]))
    plt.annotate('n = {}'.format(100), [anno_x[x], 0.6])
    

rax = plt.axes([0.3, .67, .15, .18], frameon=False)
radio = RadioButtons(rax, (5, 10, 20, 30, 40))
plt.text(-.05, 1.0, 'Bin Size', fontsize=10)
plt.tight_layout()



def animate():
    print('Animuji')

#def onClick(event):
#    global pause
#    pause ^= True
    
#plt.gcf().canvas.mpl_connect('button_press_event', onclick)
    
#a = animation.FuncAnimation(fig, update, interval=10, repeat=True, blit=False)

def run_animation():
    anim_running = True

    def onClick(event):
        nonlocal anim_running
        if anim_running:
            anim.event_source.stop()
            anim_running = False
        else:
            anim.event_source.start()
            anim_running = True

    def update(curr):
        print(curr)
        # check if animation is at the last frame, and if so, stop the animation a
        if curr == 100: 
            a.event_source.stop()

        bins1 = np.linspace(-7.5, 2.5, int(radio.value_selected))
        bins2 = np.linspace(0, 10, int(radio.value_selected))
        bins3 = np.linspace(7, 17, int(radio.value_selected))
        bins4 = np.linspace(12, 22, int(radio.value_selected))
        bins = [bins1, bins2, bins3, bins4]

        #rax = plt.axes([0.3, .67, .15, .18], frameon=False)
        #radio = RadioButtons(rax, (5, 10, 20, 30, 40))
        #plt.text(-.05, 1.0, 'Bin Size', fontsize=10)
        for x in range(0,len(axs)):
            axs[x].cla()
            axs[x].hist(eval('x'+str(x+1))[:curr*100], bins=bins[x], alpha=0.5, color=colors[x], normed=True)
            axs[x].set_title('{}'.format(title[x]))
            axs[x].annotate('n = {}'.format(curr), [anno_x[x], 0.4])
        #bins = np.arange(-4, 4, 0.5)
        #plt.hist(x[:curr], bins=bins)
        #plt.axis([-4,4,0,30])
        #plt.gca().set_title('Sampling the Normal Distribution')
        #plt.gca().set_ylabel('Frequency')
        #plt.gca().set_xlabel('Value')
        plt.tight_layout()

    fig.canvas.mpl_connect('button_press_event', onClick)

    anim = animation.FuncAnimation(fig, update)

run_animation()


radio.on_clicked(animate)

#fig = plt.figure()





# In[ ]:



