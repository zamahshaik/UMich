
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np

spam_data = pd.read_csv('spam.csv')

spam_data['target'] = np.where(spam_data['target']=='spam',1,0)
spam_data.head(10)


# In[2]:


from sklearn.model_selection import train_test_split


X_train, X_test, y_train, y_test = train_test_split(spam_data['text'], 
                                                    spam_data['target'], 
                                                    random_state=0)


# In[3]:


# Question 1

# What percentage of the documents in spam_data are spam?

# This function should return a float, the percent value (i.e. ratio∗100).

def answer_one():
    
    return len(spam_data[spam_data['target'] == 1])*100/len(spam_data)

# answer_one()


# In[4]:


# Question 2

# Fit the training data X_train using a Count Vectorizer with default parameters.

# What is the longest token in the vocabulary?

# This function should return a string.

from sklearn.feature_extraction.text import CountVectorizer

def answer_two():
    
    vect = CountVectorizer().fit(X_train)

    maxlen = max(len(word) for word in vect.get_feature_names())

    longest = sorted([(w, len(w)) for w in vect.get_feature_names() if len(w) == maxlen])[0][0]

    return longest

# answer_two()


# In[5]:


# Question 3

# Fit and transform the training data X_train using a Count Vectorizer with default parameters.

# Next, fit a fit a multinomial Naive Bayes classifier model with smoothing alpha=0.1. Find the area under the curve (AUC) score using the transformed test data.

# This function should return the AUC score as a float.

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import roc_auc_score

def answer_three():
    
    vect = CountVectorizer()
    X_train_vect = vect.fit_transform(X_train)
    X_test_vect = vect.transform(X_test)

    MultiNB = MultinomialNB(alpha = 0.1).fit(X_train_vect, y_train)

    predictions = MultiNB.predict(X_test_vect)
    AUC = roc_auc_score(y_test, predictions)
    
    return AUC

# answer_three()


# In[6]:


# Question 4

# Fit and transform the training data X_train using a Tfidf Vectorizer 
# with default parameters.

# What 20 features have the smallest tf-idf and what 20 have the largest 
# tf-idf?

# Put these features in a two series where each series is sorted by 
# tf-idf value and then alphabetically by feature name. 
# The index of the series should be the feature name, 
# and the data should be the tf-idf.

# The series of 20 features with smallest tf-idfs should be sorted 
# smallest tfidf first, the list of 20 features with largest 
# tf-idfs should be sorted largest first.

# This function should return a tuple of two series 
# (smallest tf-idfs series, largest tf-idfs series).

from sklearn.feature_extraction.text import TfidfVectorizer

def answer_four():
    vect = TfidfVectorizer()
    X_vect_trf = vect.fit_transform(X_train)

    features = vect.get_feature_names()
    tfidfs = np.array(X_vect_trf.max(0).toarray()[0])

    finarr = list(zip(features, tfidfs))

    smallest = sorted(finarr, key = lambda x: x[1])[:20]
    largest  = sorted(finarr, key = lambda x:x[1], reverse = True)[:20]

    smallest = pd.Series((x[1] for x in smallest), index = (x[0] for x in smallest))
    largest  = pd.Series((x[1] for x in largest),  index = (x[0] for x in largest))    

    return smallest, largest

# answer_four()


# In[7]:


# Question 5

# Fit and transform the training data X_train using a Tfidf Vectorizer ignoring terms that have a document 
# frequency strictly lower than 3.

# Then fit a multinomial Naive Bayes classifier model with smoothing alpha=0.1 
# and compute the area under the curve (AUC) score using the transformed test data.

# This function should return the AUC score as a float.

def answer_five():
    vect = TfidfVectorizer(min_df = 3)
    X_train_vect = vect.fit_transform(X_train)
    X_test_vect = vect.transform(X_test)

    MultiNB = MultinomialNB(alpha = 0.1).fit(X_train_vect, y_train)

    predictions = MultiNB.predict(X_test_vect)
    AUC = roc_auc_score(y_test, predictions)
        
    return AUC

# answer_five()


# In[8]:


# Question 6

# What is the average length of documents (number of characters) for not spam and spam documents?

# This function should return a tuple (average length not spam, average length spam).

def answer_six():
    spam_data['length'] = spam_data['text'].apply(lambda x:len(x))

    avgnospm = np.mean(spam_data['length'].where(spam_data['target'] == 0))
    avgspam = np.mean(spam_data['length'].where(spam_data['target'] == 1))
    
    return avgnospm, avgspam

# answer_six()


# ##### The following function has been provided to help you combine new features into the training data:

# In[9]:


def add_feature(X, feature_to_add):
    """
    Returns sparse feature matrix with added feature.
    feature_to_add can also be a list of features.
    """
    from scipy.sparse import csr_matrix, hstack
    return hstack([X, csr_matrix(feature_to_add).T], 'csr')


# In[10]:


# Question 7

# Fit and transform the training data X_train using a Tfidf Vectorizer 
# ignoring terms that have a document frequency strictly lower than 5.

# Using this document-term matrix and an additional feature, 
# the length of document (number of characters), 
# fit a Support Vector Classification model with regularization C=10000. 
# Then compute the area under the curve (AUC) score using the transformed test 
# data.

# This function should return the AUC score as a float.
from sklearn.svm import SVC

def answer_seven():
 
    vect = TfidfVectorizer(min_df = 5)

    X_train_vect = vect.fit_transform(X_train)
    X_train_adfeat = add_feature(X_train_vect, X_train.str.len())

    X_test_vect = vect.transform(X_test)
    X_test_adfeat = add_feature(X_test_vect, X_test.str.len())

    svcla = SVC(C = 10000).fit(X_train_adfeat, y_train)

    predictions = svcla.predict(X_test_adfeat)

    auc = roc_auc_score(y_test, predictions)

    return auc

# answer_seven()


# In[11]:


# Question 8

# What is the average number of digits per document for not spam and spam documents?

# This function should return a tuple (average # digits not spam, average # digits spam).

def answer_eight():

    spam_data['length'] = spam_data['text'].apply(lambda x: len([word for word in x if word.isdigit()]))

    avgnospm = np.mean(spam_data['length'].where(spam_data['target'] == 0))
    avgspam = np.mean(spam_data['length'].where(spam_data['target'] == 1))


    
    return avgnospm, avgspam

# answer_eight()


# In[12]:


# Question 9

# Fit and transform the training data X_train using a Tfidf Vectorizer ignoring terms that have a document frequency 
# strictly lower than 5 and using word n-grams from n=1 to n=3 (unigrams, bigrams, and trigrams).

# Using this document-term matrix and the following additional features:

#     1. the length of document (number of characters)
#     2. number of digits per document

# fit a Logistic Regression model with regularization C=100. 
# Then compute the area under the curve (AUC) score using the transformed test data.

# This function should return the AUC score as a float.

from sklearn.linear_model import LogisticRegression

def answer_nine():
    
    vect = TfidfVectorizer(min_df = 5, ngram_range = [1, 3])

    X_train_vect = vect.fit_transform(X_train)
    X_train_adfeat = add_feature(X_train_vect, [X_train.str.len(), 
                                                X_train.apply(lambda x: len([word for word in x if word.isdigit()]))])

    X_test_vect = vect.transform(X_test)
    X_test_adfeat = add_feature(X_test_vect, [X_test.str.len(),
                                              X_test.apply(lambda x: len([word for word in x if word.isdigit()]))])

    logreg = LogisticRegression(C = 100).fit(X_train_adfeat, y_train)

    predictions = logreg.predict(X_test_adfeat)

    auc = roc_auc_score(y_test, predictions)
        
    return auc

# answer_nine()


# In[13]:


# Question 10

# What is the average number of non-word characters (anything other than a letter, digit or underscore) per document 
# for not spam and spam documents?

# Hint: Use \w and \W character classes

# This function should return a tuple (average # non-word characters not spam, average # non-word characters spam).

def answer_ten():

    spam_data['length'] = spam_data['text'].str.findall(r'\W').str.len()

    avgnospm = np.mean(spam_data['length'].where(spam_data['target'] == 0))
    avgspam = np.mean(spam_data['length'].where(spam_data['target'] == 1))
    
    return avgnospm, avgspam

# answer_ten()


# In[14]:


# Question 11

# Fit and transform the training data X_train using a Count Vectorizer ignoring terms that have a 
# document frequency strictly lower than 5 and using character n-grams from n=2 to n=5.

# To tell Count Vectorizer to use character n-grams pass in analyzer='char_wb' which creates character 
# n-grams only from text inside word boundaries. This should make the model more robust to spelling mistakes.

# Using this document-term matrix and the following additional features:

#     the length of document (number of characters)
#     number of digits per document
#     number of non-word characters (anything other than a letter, digit or underscore.)

# fit a Logistic Regression model with regularization C=100. 
# Then compute the area under the curve (AUC) score using the transformed test data.

# Also find the 10 smallest and 10 largest coefficients from the model 
# and return them along with the AUC score in a tuple.

# The list of 10 smallest coefficients should be sorted smallest first, the list of 10 largest coefficients 
# should be sorted largest first.

# The three features that were added to the document term matrix should have the following names 
# should they appear in the list of coefficients: ['length_of_doc', 'digit_count', 'non_word_char_count']

# This function should return a tuple (AUC score as a float, smallest coefs list, largest coefs list).

def answer_eleven():

    vect = CountVectorizer(min_df = 5, ngram_range = [2, 5], analyzer='char_wb')

    X_train_vect = vect.fit_transform(X_train)
    X_train_adfeat = add_feature(X_train_vect, [X_train.str.len(), 
                                                X_train.apply(lambda x: len([word for word in x if word.isdigit()])),
                                                X_train.str.findall(r'\W').str.len()])

    X_test_vect = vect.transform(X_test)
    X_test_adfeat = add_feature(X_test_vect, [X_test.str.len(),
                                              X_test.apply(lambda x: len([word for word in x if word.isdigit()])),
                                              X_test.str.findall(r'\W').str.len()])

    logreg = LogisticRegression(C = 100).fit(X_train_adfeat, y_train)

    predictions = logreg.predict(X_test_adfeat)

    auc = roc_auc_score(y_test, predictions)

    feature_names = np.array(vect.get_feature_names() + ['length_of_doc', 'digit_count', 'non_word_char_count'])

    sorted_coef_index = logreg.coef_[0].argsort()

    smallest = feature_names[sorted_coef_index[:10]]
    largest = feature_names[sorted_coef_index[:-11:-1]]

    return (auc, smallest, largest)

# answer_eleven()

