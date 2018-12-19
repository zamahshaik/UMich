
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

def blight_model():

    train_df = pd.read_csv('train.csv', encoding = 'ISO-8859-1')

    # pd.set_option('display.max_columns', None)

    # Remove rows that don't have a complaince value
    train_df = train_df[train_df['compliance'].notnull()]

    # Remove columns to prevent data leakage
    leak_cols = ['payment_amount', 'balance_due', 'payment_date', 'payment_status', 'collection_status', 'compliance_detail']
    train_df = train_df.drop(train_df[leak_cols], axis = 1)

    train_df = train_df.set_index('ticket_id')

    col_list = ['disposition', 'fine_amount', 'admin_fee', 'state_fee', 'late_fee', 'discount_amount', 
                'clean_up_cost', 'judgment_amount', 'compliance']

    train_df = train_df[col_list]

    # When disposition is removed the AUC fall from 0.7973 to 0.7616. Converting disposition from label to integer

    label_encoder = LabelEncoder()
    train_df['disposition'] = label_encoder.fit_transform(train_df['disposition'])
    train_df

    X = train_df[train_df.columns[:-1]]
    y = train_df[train_df.columns[-1]]

    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state = 0)

    clf = DecisionTreeClassifier()
    clf.fit(X_train, y_train)
    predictions = clf.predict_proba(X_test)
    print('Test set AUC: ', roc_auc_score(y_test, predictions[:,1]))

    test_df = pd.read_csv('test.csv', encoding = 'ISO-8859-1')

    test_df = test_df.set_index('ticket_id')

    col_list = ['disposition', 'fine_amount', 'admin_fee', 'state_fee', 'late_fee', 'discount_amount', 
                'clean_up_cost', 'judgment_amount']

    test_df = test_df[col_list]

    # When disposition is removed the AUC fall from 0.7973 to 0.7616. Converting disposition from label to integer

    label_encoder = LabelEncoder()
    test_df['disposition'] = label_encoder.fit_transform(test_df['disposition'])

    clf.fit(X_train, y_train)

    predictions = pd.DataFrame(clf.predict_proba(test_df), index = test_df.index)

    compliance_test = predictions[1].rename('compliance').astype('float32')

    return compliance_test

