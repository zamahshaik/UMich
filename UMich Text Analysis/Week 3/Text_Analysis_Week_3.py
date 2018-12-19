
# coding: utf-8

# # Case Study: Sentiment Analysis

# ### Data Prep

# In[1]:


import numpy as np
import pandas as pd

df = pd.read_csv('Amazon_Unlocked_Mobile.csv')

# Sample the data to speed up computation
# df = df.sample(frac = 0.1, random_state = 10)

df.head()


# In[2]:


# drop missing values
df.dropna(inplace = True)

# Remove neutral ratings equal to 3
df = df[df['Rating'] != 3]

# Encode 4s and 5s as 1 (rated positively)
# Encode 1s and 2s as 0 (rated poorly)
df['Positively Rated'] = np.where(df['Rating'] > 3, 1, 0)
df.head(10)


# In[3]:


# Most ratings are positive
df['Positively Rated'].mean()


# In[4]:


from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(df['Reviews'],
                                                    df['Positively Rated'],
                                                    random_state = 0)


# In[5]:


print('X_train first entry:\n\n', X_train.iloc[0])
print('\n\nX_train shape: ', X_train.shape)


# ## Count Vectorizer

# In[6]:


from sklearn.feature_extraction.text import CountVectorizer

# Fit the CountVectorizer to the training data
vect = CountVectorizer().fit(X_train)


# In[7]:


vect.get_feature_names()[::2000]


# In[8]:


len(vect.get_feature_names())


# In[9]:


# transform the docs in training data to a doc-term matrix
X_train_vectorized = vect.transform(X_train)

X_train_vectorized


# In[10]:


from sklearn.linear_model import LogisticRegression

# train the model
model = LogisticRegression()
model.fit(X_train_vectorized, y_train)


# In[11]:


from sklearn.metrics import roc_auc_score

# predict the transformed test docs
predictions = model.predict(vect.transform(X_test))

print('AUC: ', roc_auc_score(y_test, predictions))


# In[12]:


# get the feature names as numpy array
feature_names = np.array(vect.get_feature_names())

# sort the coefficients from the model
sorted_coef_index = model.coef_[0].argsort()

# Find the 10 smallest and 10 largest coeffs
# the 10 largest coefs are being indexed using [:-11, -1]
# so the list returned is in order of largest to smallest
print('Smallest Coefs:\n{}\n'.format(feature_names[sorted_coef_index[:10]]))
print('Largest coefs: \n{}'.format(feature_names[sorted_coef_index[:-11:-1 ]]))


# ## Tfidf

# In[13]:


from sklearn.feature_extraction.text import TfidfVectorizer

# Fit the TfidfVectorizer to the training data specifiying a min doc freq of 5
vect = TfidfVectorizer(min_df = 5).fit(X_train)
len(vect.get_feature_names())


# In[14]:


X_train_vectorized = vect.transform(X_train)

model = LogisticRegression()
model.fit(X_train_vectorized, y_train)

predictions = model.predict(vect.transform(X_test))

print('AUC: ', roc_auc_score(y_test, predictions))


# In[15]:


feature_names = np.array(vect.get_feature_names())

sorted_tfidf_index = X_train_vectorized.max(0).toarray()[0].argsort()

print('Smallest tfidf:\n{}\n'.format(feature_names[sorted_tfidf_index[:10]]))
print('Largest tfidf: \n{}'.format(feature_names[sorted_tfidf_index[:-11:-1]]))


# In[16]:


sorted_coef_index = model.coef_[0].argsort()

print('Smallest coef:\n{}\n'.format(feature_names[sorted_coef_index[:10]]))
print('Largest coefs:\n{}'.format(feature_names[sorted_coef_index[:-11:-1]]))


# In[17]:


# These reviews are treated the same by our current model
print(model.predict(vect.transform(['not an issue, phone is working',
                                    'an issue, phone is not working'])))


# ## n-grams

# In[18]:


# Fit the CountVectorizer to the training data specifiying a min
# doc freq of 5 and extracting 1-grams and 2-grams

vect = CountVectorizer(min_df = 5, ngram_range = (1,2)).fit(X_train)

X_train_vectorized = vect.transform(X_train)

len(vect.get_feature_names())


# In[19]:


model = LogisticRegression()
model.fit(X_train_vectorized, y_train)

predictions = model.predict(vect.transform(X_test))

print('AUC: ', roc_auc_score(y_test, predictions))


# In[20]:


feature_names = np.array(vect.get_feature_names())

sorted_coef_index = model.coef_[0].argsort()

print('Smallest Coefs:\n{}\n'.format(feature_names[sorted_coef_index[:10]]))
print('Largest Coefs: \n{}'.format(feature_names[sorted_coef_index[:-11:-1]]))


# In[21]:


# These reviews are now correctly identified

print(model.predict(vect.transform(['not an issue, phone is working',
                                   'an issue, phone is not working'])))

