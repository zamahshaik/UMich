
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np

train_df = pd.read_csv('train.csv',encoding = 'ISO-8859-1',low_memory=False)


# In[2]:


train_df.head()


# In[3]:


# drop rows with null values in the 'compliance' target column
train_df = train_df.dropna(subset=['compliance'])

train_df.head()


# In[4]:


# drop train set only columns except for the 'compliance' target column
drop_columns0 = ['disposition', 'payment_amount','payment_date','payment_status',
                 'balance_due','collection_status','compliance_detail']
train_df = train_df.drop(train_df[drop_columns0],axis=1)

train_df.head()


# In[5]:


train_df.info()


# In[6]:


# fillna
train_df = train_df.fillna(-1)

train_df.head()


# In[7]:


keep_columns_train = ['fine_amount','admin_fee','state_fee',
                      'late_fee','discount_amount','clean_up_cost',
                      'judgment_amount',
                      'compliance']
train_df = train_df[keep_columns_train]


# In[8]:


train_df


# In[8]:


from sklearn.preprocessing import LabelEncoder

# integer encode
label_encoder = LabelEncoder()
train_df['disposition'] = label_encoder.fit_transform(train_df['disposition'])
train_df


# In[10]:


# convert_catetorical = set(train_df['disposition'])|{'<unknown>'}
# train_df['disposition']= (pd.Categorical(train_df['disposition'],
#                          categories=convert_catetorical).fillna('<unknown>').codes)


# In[9]:


train_df.columns


# In[10]:


train_df


# In[11]:


X = train_df[train_df.columns[:-1]]
y = train_df[train_df.columns[-1]]


# In[12]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

## cross validation for better model

from sklearn.model_selection import cross_val_score
from sklearn.ensemble import GradientBoostingClassifier
gradient_boost_clf = GradientBoostingClassifier()
cv_scores = cross_val_score(gradient_boost_clf,X,y,cv=5)
cv_scores


# In[13]:


## gridsearch for better parameters

from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import GridSearchCV
gradient_boost_clf = GradientBoostingClassifier()
grid_values = {'n_estimators':[10,30,50],
              'learning_rate':[0.01,0.1,1],
              'max_depth':[3,4,5]}
grid_auc = GridSearchCV(gradient_boost_clf,param_grid=grid_values,scoring='roc_auc')
grid_auc.fit(X_train,y_train)


# In[14]:


from sklearn.metrics import roc_auc_score
y_decision_scores_auc = grid_auc.decision_function(X_test) 
print('test set AUC:', roc_auc_score(y_test, y_decision_scores_auc))
print('grid best paramete max AUC:', grid_auc.best_params_)
print('grid best score AUC: ', grid_auc.best_score_)

## better model and parameters

gradient_boost_clf = GradientBoostingClassifier(learning_rate=0.1,
                                               max_depth=5,
                                               n_estimators=30)


# In[19]:


## test set predict and predict_proba

test_df = pd.read_csv('test.csv')
test_df = test_df.fillna(-1)
keep_columns_test = ['ticket_id','disposition','fine_amount','admin_fee','state_fee',
                      'late_fee','discount_amount','clean_up_cost','judgment_amount',]
test_df = test_df[keep_columns_test]
test_df = test_df.set_index('ticket_id')

label_encoder1 = LabelEncoder()
test_df['disposition'] = label_encoder1.fit_transform(test_df['disposition'])


# In[20]:


test_df


# In[17]:


gradient_boost_clf.fit(X_train,y_train)
# predict on test set'
y_predict = gradient_boost_clf.predict(test_df)
# predict probability on test set'
y_proba = gradient_boost_clf.predict_proba(test_df)

y_proba_df = pd.DataFrame(y_proba,index=test_df.index)

y_proba_compliance = y_proba_df[1].rename('compliance').astype('float32')


# In[18]:


y_proba_compliance


# In[21]:


train_df['disposition'].unique()


# In[22]:


test_df['disposition'].unique()

