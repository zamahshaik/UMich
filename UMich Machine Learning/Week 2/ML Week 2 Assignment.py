
# coding: utf-8

# In[14]:


import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

np.random.seed(0)
n = 15
x = np.linspace(0,10,n) + np.random.randn(n)/5
y = np.sin(x)+x/6 + np.random.randn(n)/10

X_train, X_test, y_train, y_test = train_test_split(x, y, random_state=0)

# You can use this function to help you visualize the dataset by
# plotting a scatterplot of the data points
# in the training and test sets.
# def part1_scatter():
#     import matplotlib.pyplot as plt
#     %matplotlib notebook
#     plt.figure()
#     plt.scatter(X_train, y_train, label='training data')
#     plt.scatter(X_test, y_test, label='test data')
#     plt.legend(loc=4);
#     plt.show()    
# NOTE: Uncomment the function below to visualize the data, but be sure 
# to **re-comment it before submitting this assignment to the autograder**.   
# part1_scatter() 


# In[2]:


def answer_one():
    from sklearn.linear_model import LinearRegression
    from sklearn.preprocessing import PolynomialFeatures
    
    np.random.seed(0)
    n = 15
    x = np.linspace(0,10,n) + np.random.randn(n)/5
    y = np.sin(x)+x/6 + np.random.randn(n)/10
    
    x = x.reshape(-1,1)

    y = y.reshape(-1, 1)

    X_predict_input = np.linspace(0, 10, 100).reshape(-1, 1)

    degree_predictions = []

    for d in [1, 3, 6, 9]:
        poly = PolynomialFeatures(degree = d)
        X_poly = poly.fit_transform(x)
        X_100 = poly.fit_transform(X_predict_input)

        X_train, X_test, y_train, y_test = train_test_split(X_poly, y, random_state = 0)

        linreg = LinearRegression().fit(X_train, y_train)
        degree_predictions = np.append(degree_predictions, linreg.predict(X_100))
                  
    return degree_predictions.reshape(4, 100)


# In[6]:


# feel free to use the function plot_one() to replicate the figure 
# from the prompt once you have completed question one
# def plot_one(degree_predictions):
    
#     import matplotlib.pyplot as plt
#     %matplotlib notebook

#     plt.figure(figsize=(10,5))
#     plt.plot(X_train, y_train, 'o', label='training data', markersize=10)
#     plt.plot(X_test, y_test, 'o', label='test data', markersize=10)
#     for i,degree in enumerate([1,3,6,9]):
#         plt.plot(np.linspace(0, 10, 100), degree_predictions[i], alpha=0.8, lw = 2, label='degree={}'.format(degree))
#     plt.ylim(-1,2.5)
#     plt.legend(loc=4)
#     plt.show()

# plot_one(answer_one())


# In[7]:


def answer_two():
    from sklearn.linear_model import LinearRegression
    from sklearn.preprocessing import PolynomialFeatures
    from sklearn.metrics.regression import r2_score

    np.random.seed(0)
    n = 15
    x = np.linspace(0,10,n) + np.random.randn(n)/5
    y = np.sin(x)+x/6 + np.random.randn(n)/10    
    
    r2_train = []
    r2_test = []
    
    x = x.reshape(-1, 1)
    
    for d in [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]:
    
        polyn = PolynomialFeatures(degree = d)
        X_Poly = polyn.fit_transform(x)
        X_train, X_test, y_train, y_test = train_test_split(X_Poly, y, random_state = 0)

        linregr = LinearRegression().fit(X_train, y_train)

        r2_train_score = linregr.score(X_train, y_train)
        r2_test_score  = linregr.score(X_test, y_test)
        
        r2_train = np.append(r2_train, [r2_train_score], axis = 0)
        r2_test  = np.append(r2_test, [r2_test_score], axis = 0)
        
    return r2_train, r2_test


# In[8]:


def answer_three():
        
#     import matplotlib.pyplot as plt
#     %matplotlib notebook

    score_train, score_test = answer_two()

#     plt.figure(figsize=(10,5))

#     # for degree in zip([0, 1, 2, 3, 4, 5, 6, 7, 8, 9]):

#     #     plt.plot(degree, r_train, 'o', label='training data', markersize=10)
#     plt.plot([0, 1, 2, 3, 4, 5, 6, 7, 8, 9], score_train, 'o', label = 'train data', markersize = 5)
#     plt.plot([0, 1, 2, 3, 4, 5, 6, 7, 8, 9], score_test, 'o', label='test data', markersize=10)

#     #     for i,degree in enumerate([1,3,6,9]):
#     #         plt.plot(np.linspace(0, 10, 100), degree_predictions[i], alpha=0.8, lw = 2, label='degree={}'.format(degree))
#     #     plt.ylim(-1,2.5)
#     plt.legend(loc=2)
#     plt.xlabel('Degree')
#     plt.ylabel('Score')
#     plt.show()

    x = (1, 9, 6)

    return x


# In[9]:


def answer_four():
    from sklearn.preprocessing import PolynomialFeatures
    from sklearn.linear_model import Lasso, LinearRegression
    from sklearn.metrics.regression import r2_score

    np.random.seed(0)
    n = 15
    x = np.linspace(0,10,n) + np.random.randn(n)/5
    y = np.sin(x)+x/6 + np.random.randn(n)/10    

    r2_train = []
    r2_test = []

    x = x.reshape(-1, 1)

    polyn = PolynomialFeatures(degree = 12)
    X_Poly = polyn.fit_transform(x)
    X_train, X_test, y_train, y_test = train_test_split(X_Poly, y, random_state = 0)

    linregr = LinearRegression().fit(X_train, y_train)

    r2_train_score = linregr.score(X_train, y_train)
    LinearRegression_R2_test_score  = linregr.score(X_test, y_test)

    lassolin = Lasso(alpha = 0.01, max_iter = 10000).fit(X_train, y_train)

    Lasso_R2_test_score = lassolin.score(X_test, y_test)

    return (LinearRegression_R2_test_score, Lasso_R2_test_score)


# In[10]:


import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split


mush_df = pd.read_csv('mushrooms.csv')
mush_df2 = pd.get_dummies(mush_df)

X_mush = mush_df2.iloc[:,2:]
y_mush = mush_df2.iloc[:,1]

# use the variables X_train2, y_train2 for Question 5
X_train2, X_test2, y_train2, y_test2 = train_test_split(X_mush, y_mush, random_state=0)

# For performance reasons in Questions 6 and 7, we will create a smaller version of the
# entire mushroom dataset for use in those questions.  For simplicity we'll just re-use
# the 25% test split created above as the representative subset.
#
# Use the variables X_subset, y_subset for Questions 6 and 7.
X_subset = X_test2
y_subset = y_test2


# In[11]:


def answer_five():
    from sklearn.tree import DecisionTreeClassifier

    clf = DecisionTreeClassifier(random_state = 0).fit(X_train2, y_train2)

    l1 = (pd.DataFrame(clf.feature_importances_, X_train2.columns, columns = ['Importance'])
            .sort_values(by = 'Importance', ascending = False)[0:5])

    l2 = list(l1.index)

    return l2


# In[12]:


def answer_six():
    from sklearn.svm import SVC
    from sklearn.model_selection import validation_curve

    param_range = np.logspace(-4, 1, 6)

    clf = SVC(kernel = 'rbf', C = 1, random_state = 0)

    train_scores, test_scores = validation_curve(SVC(), X_subset, y_subset,
                                                 param_name = 'gamma',
                                                 param_range = param_range, 
                                                 scoring = 'accuracy')

    train_mean = train_scores.mean(axis = 1)
    test_mean = test_scores.mean(axis = 1)

    return train_mean, test_mean


# In[13]:


def answer_seven():
    
#     import matplotlib.pyplot as plt
#     %matplotlib notebook
    from sklearn.svm import SVC
    from sklearn.model_selection import validation_curve

    param_range = np.logspace(-4, 1, 6)

    clf = SVC(kernel = 'rbf', C = 1, random_state = 0)

    train_scores, test_scores = validation_curve(SVC(), X_subset, y_subset,
                                                 param_name = 'gamma',
                                                 param_range = param_range, 
                                                 scoring = 'accuracy')

#     plt.figure(figsize=(10,5))

#     plt.plot(np.logspace(-4, 1, 6), train_scores, 'o', label = 'train data', markersize = 5, color = 'b')
#     plt.plot(np.logspace(-4, 1, 6), test_scores,  'o', label='test data', markersize=10, color = 'r')

#     plt.xlabel('Gamma')
#     plt.ylabel('Score')
#     plt.show();
    
    
    
    return (0.001, 10.0, 0.1)

