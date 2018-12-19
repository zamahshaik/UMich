
# coding: utf-8

# In[1]:


get_ipython().run_line_magic('matplotlib', 'notebook')
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

fruits = pd.read_table('fruit_data_with_colors.txt')


# In[2]:


fruits.head()


# In[3]:


lookup_fruit_name = dict(zip(fruits.fruit_label.unique(), fruits.fruit_name.unique()))
lookup_fruit_name


# # Create Train-Test Split

# In[4]:


X = fruits[['mass', 'width', 'height', 'color_score']]
y = fruits['fruit_label']

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state = 0) #Standard 75-25% split


# # Visualization

# In[6]:


from matplotlib import cm

cmap = cm.get_cmap('gnuplot')

scatter = pd.scatter_matrix(X_train, c= y_train, marker = 'o', s = 40, hist_kwds = {'bins': 15}, 
                            figsize = (10, 6), cmap = cmap)


# In[7]:


from mpl_toolkits.mplot3d import Axes3D

fig = plt.figure()
ax = fig.add_subplot(111, projection = '3d')
ax.scatter(X_train['width'], X_train['height'], X_train['color_score'], c = y_train, marker = 'o', s = 100)
ax.set_xlabel('width')
ax.set_ylabel('height')
ax.set_zlabel('color_score')
plt.show()


# # kNN k Nearest Neighbors Classifier

# In[9]:


# kNN is a instance or memory based supervised learning method. 
# Memorize the labelled examples and use that to identify new data
# k is number of nearest neighbors the classifier will retrieve.
# Feature Space, Decision Boundary

'''Given a training set X_train with labels y_train and a new instance x_test to be classified:
1. find most similar instances (say X_NN) to x_test that are in X_train.
2. get the labels y_NN for instances X_NN
3. predict the label for x_tet by combining the labels y_NN (eg: Simple majority vote)

4 things are needed for the nearest neighbor algo:
1. distance metric (eg: Euclidean distance Minkowski with p = 2)
2. how man nearest neighbors to look at? ie value of k (usually odd)
3. optional weighting function on the neighboring points
4. method for aggregating the classes of neighboring points (simple majority vote)
'''


# # create classifier object

# In[10]:


from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors = 5)


# # Train the classifier (fit the estimator) using the training data

# In[12]:


knn.fit(X_train, y_train)


# # Estimate the accuracy of the classifier on future data, using the test data

# In[13]:


knn.score(X_test, y_test)


# # Use the trained kNN classifier model to classify new, unseen objects

# In[27]:


fruit_prediction = knn.predict([[20, 4.3, 5.5, 0.65]])
lookup_fruit_name[fruit_prediction[0]]


# In[16]:


fruit_prediction = knn.predict([[100, 6.3, 8.5, 0.45]])
lookup_fruit_name[fruit_prediction[0]]


# # Plot decision boundaries of the kNN classifier

# In[24]:


from adspy_shared_utilities import plot_fruit_knn
plot_fruit_knn(X_train, y_train, 6, 'uniform') # uniform is the weighting method.


# In[23]:


plot_fruit_knn(X_train, y_train, 6, 'distance')


# In[22]:


k_range = range(1, 20)
scores = []

for k in k_range:
    knn = KNeighborsClassifier(n_neighbors = k)
    knn.fit(X_train, y_train)
    scores.append(knn.score(X_test, y_test))
    
plt.figure()
plt.xlabel('k')
plt.ylabel('accuracy')
plt.scatter(k_range, scores)
plt.xticks([0, 5, 10, 15, 20]);

