
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd

df = pd.read_csv('fraud_data.csv')
# df.head()


# In[3]:


def answer_one():
    percent = df[df['Class'] == 1]['Class'].count()/df['Class'].count()
    return percent


# In[4]:


answer_one()


# In[5]:


from sklearn.model_selection import train_test_split

df = pd.read_csv('fraud_data.csv')

X = df.iloc[:,:-1]
y = df.iloc[:,-1]

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)


# In[6]:


def answer_two():
    from sklearn.dummy import DummyClassifier
    from sklearn.metrics import recall_score
    from sklearn.metrics import accuracy_score

    dummy = DummyClassifier(strategy = 'most_frequent').fit(X_train, y_train)
    dummy_pred = dummy.predict(X_test)

    accuracy_score = accuracy_score(y_test, dummy_pred)
    recall_score = recall_score(y_test, dummy_pred)
    
    return (accuracy_score, recall_score)


# In[7]:


# Using X_train, X_test, y_train, y_test (as defined above), 
# train a SVC classifer using the default parameters. 
# What is the accuracy, recall, and precision of this classifier?

# This function should a return a tuple with three floats, i.e. (accuracy score, recall score, precision score).

def answer_three():
    from sklearn.metrics import recall_score, precision_score
    from sklearn.svm import SVC

    svm = SVC().fit(X_train, y_train)
    svm_pred = svm.predict(X_test)

    acc = svm.score(X_test, y_test)
    recall_score = recall_score(y_test, svm_pred)
    precision_score = precision_score(y_test, svm_pred)
    
    return (acc, recall_score, precision_score)


# In[8]:


# Using the SVC classifier with parameters {'C': 1e9, 'gamma': 1e-07}, 
# what is the confusion matrix when using a threshold of -220 on the decision function. 
# Use X_test and y_test.
# This function should return a confusion matrix, a 2x2 numpy array with 4 integers.

def answer_four():
    from sklearn.metrics import confusion_matrix
    from sklearn.svm import SVC

    svm = SVC(C = 1e9, gamma = 1e-07).fit(X_train, y_train)
    svm_pred = svm.decision_function(X_test) > -220

    confusion = confusion_matrix(y_test, svm_pred)
    
    return confusion


# In[11]:


from sklearn.metrics import confusion_matrix
from sklearn.svm import SVC

svm = SVC(C = 1e9, gamma = 1e-07).fit(X_train, y_train)
svm_pred = svm.decision_function(X_test) > -220

confusion = confusion_matrix(y_test, svm_pred)

print(confusion)


# In[23]:


# Train a logisitic regression classifier with default parameters using X_train and y_train.

# For the logisitic regression classifier, create a precision recall curve and a roc curve using y_test 
# and the probability estimates for X_test (probability it is fraud).

# Looking at the precision recall curve, what is the recall when the precision is 0.75?

# Looking at the roc curve, what is the true positive rate when the false positive rate is 0.16?

# This function should return a tuple with two floats, i.e. (recall, true positive rate).

def answer_five():
    
#     %matplotlib notebook
#     import seaborn as sns
#     import matplotlib.pyplot as plt
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import precision_recall_curve, roc_curve, auc

    lr = LogisticRegression().fit(X_train, y_train)
    lr_pred = lr.predict_proba(X_test)

    lr_pred1 = lr.decision_function(X_test)

    precision, recall, threshold = precision_recall_curve(y_test, lr_pred1)
    closest_zero = np.argmin(np.abs(threshold))
    closest_zero_p = precision[closest_zero]
    closest_zero_r = recall[closest_zero]

#     plt.figure()
#     plt.xlim([0.0, 1.01])
#     plt.ylim([0.0, 1.01])
#     plt.plot(precision, recall, label = 'Precision-Recall Curve')
#     plt.plot(closest_zero_p, closest_zero_r, 'o', markersize = 12, fillstyle = 'none', c = 'r', mew = 3)
#     plt.xlabel('Precision', fontsize = 16)
#     plt.ylabel('Recall', fontsize = 16)
#     plt.axes().set_aspect('equal')

    fpr, tpr, _ = roc_curve(y_test, lr_pred1)
    roc_auc_lr = auc(fpr, tpr)

#     plt.figure()
#     plt.xlim([-0.01, 1.00])
#     plt.ylim([-0.01, 1.01])
#     plt.plot(fpr, tpr, lw = 3, label = 'LogRegr ROC curve (area = {:0.2f})'.format(roc_auc_lr))
#     plt.xlabel('False Positive rate', fontsize = 16)
#     plt.ylabel('True Positive rate', fontsize = 16)
#     plt.title('ROC curve (Credit Card Fraud)', fontsize = 16)
#     plt.legend(loc = 'lower right', fontsize = 13)
#     plt.plot([0, 1], [0, 1], color = 'navy', lw = 3, linestyle = '--')
#     plt.axes().set_aspect('equal')
#     plt.show()
    
    return (0.825, 0.947)


# In[24]:


answer_five()


# In[33]:


# Perform a grid search over the parameters listed below for a Logisitic Regression classifier, 
# using recall for scoring and the default 3-fold cross validation.
# 'penalty': ['l1', 'l2']
# 'C':[0.01, 0.1, 1, 10, 100]
# From .cv_results_, create an array of the mean test scores of each parameter combination. i.e.
# This function should return a 5 by 2 numpy array with 10 floats.
# Note: do not return a DataFrame, just the values denoted by '?' above in a numpy array. 
# You might need to reshape your raw result to meet the format we are looking for.

def answer_six():    
    from sklearn.model_selection import GridSearchCV
    from sklearn.linear_model import LogisticRegression

    lr = LogisticRegression()

    grid_values = {'penalty': ['l1', 'l2'], 'C': [0.01, 0.1, 1, 10, 100]}

    grid_srch = GridSearchCV(lr, param_grid = grid_values, scoring = 'recall').fit(X_train, y_train)

    arr1 = grid_srch.cv_results_['mean_test_score'].reshape(5, 2)
    
    return arr1


# In[34]:


# Use the following function to help visualize results from the grid search
# def GridSearch_Heatmap(scores):
#     %matplotlib notebook
#     import seaborn as sns
#     import matplotlib.pyplot as plt
#     plt.figure()
#     sns.heatmap(scores.reshape(5,2), xticklabels=['l1','l2'], yticklabels=[0.01, 0.1, 1, 10, 100])
#     plt.yticks(rotation=0);

# GridSearch_Heatmap(answer_six())

