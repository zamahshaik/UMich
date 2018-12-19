
# coding: utf-8

# In[8]:


import pandas as pd
import numpy as np
import re

def date_sorter():
    
    doc = []
    with open('dates.txt') as file:
        for line in file:
            doc.append(line)
    df = pd.Series(doc)
    # Dates in Regular Format - mm/dd/yyyy, mm/d/yy, mm/d/yyyy, m/d/yy, m/d/yyyy, mm/d/yyyy...
    re1 = df.str.extractall(r'(?:(?P<Month>\d{1,2})[/-](?P<Day>\d{1,2})[/-](?P<Year>(?:19|20)?\d{2}))')
    re1['Month'] = re1['Month'].apply(lambda x: '0'+x if len(x) < 2 else x)
    re1['Day'] = re1['Day'].apply(lambda x: '0'+x if len(x) < 2 else x)
    re1['Year'] = re1['Year'].apply(lambda x: '19'+x if len(x) < 4 else x)
    re1 = re1[re1.Day.astype(int) < 32]
    re1 = re1[re1.Month.astype(int) < 13]

    df1 = pd.DataFrame((re1['Month']+'/'+re1['Day']+'/'+re1['Year']).astype('datetime64'), columns = ['Date'])
    df1.reset_index(inplace = True)
    df1.drop(['match'], axis = 1, inplace = True)
    df1 = df1.rename(columns={'level_0': 'Old_Index', 'Date': 'Date'})

    # Dates in dd Mon yyyy, dd Month yyyy format
    re2 = df.str.extractall(r'(?:(?P<Day>\d{2} )(?P<Month>(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z.,/ ]*) (?P<Year>(?:19|20)?\d{2}))')
    months = ({'January': '01', 'February': '02', 'March': '03', 'April': '04', 'May': '05', 'June': '06', 
              'July': '07', 'August': '08', 'September': '09', 'October': '10', 'November': '11', 'December': '12', 
              'Decemeber': '12', 'Janaury': '01',
              'Jan': '01', 'Feb': '02', 'Mar': '03', 'Apr': '04', 'May': '05', 'Jun': '06', 
              'Jul': '07', 'Aug': '08', 'Sep': '09', 'Oct': '10', 'Nov': '11', 'Dec': '12'})

    re2['Month'] = re2['Month'].map(months)
    re2['Day'] = re2['Day'].apply(lambda x: '0'+x if len(x) < 2 else x)
    re2['Year'] = re2['Year'].apply(lambda x: '19'+x if len(x) < 4 else x)
    re2 = re2[re2.Day.astype(int) < 32]
    re2 = re2[re2.Month.astype(int) < 13]

    df2 = pd.DataFrame((re2['Month']+'/'+re2['Day']+'/'+re2['Year']).astype('datetime64'), columns = ['Date'])
    df2.reset_index(inplace = True)
    df2.drop(['match'], axis = 1, inplace = True)
    df2 = df2.rename(columns={'level_0': 'Old_Index', 'Date': 'Date'})

    # Dates in Mon dd, yyyy, Month dd, yyyy, Month. dd, yyyy, Mon. dd yyyy format
    re3 = df.str.extractall(r'(?:(?P<Month>(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z. ]*)(?P<Day>\d{2}[/., ] )(?P<Year>(?:19|20)?\d{2}))')
    re3['Month'] = re3['Month'].str.replace(".", "").str.strip()
    re3['Day'] = re3['Day'].str.replace(",", "")

    re3['Month'] = re3['Month'].map(months)
    re3['Day'] = re3['Day'].apply(lambda x: '0'+x if len(x) < 2 else x)
    re3['Year'] = re3['Year'].apply(lambda x: '19'+x if len(x) < 4 else x)
    re3 = re3[re3.Day.astype(int) < 32]
    re3 = re3[re3.Month.astype(int) < 13]

    df3 = pd.DataFrame((re3['Month']+'/'+re3['Day']+'/'+re3['Year']).astype('datetime64'), columns = ['Date'])
    df3.reset_index(inplace = True)
    df3.drop(['match'], axis = 1, inplace = True)
    df3 = df3.rename(columns={'level_0': 'Old_Index', 'Date': 'Date'})

    # Dates in Mon dd yyyy, Month dd yyyy
    re4 = df.str.extractall(r'(?:(?P<Month>(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z. ]*)(?P<Day>\d{2} )(?P<Year>(?:19|20)?\d{2}))')    

    re4['Month'] = re4['Month'].str.replace(".", "").str.strip()
    re4['Day'] = re4['Day'].str.replace(",", "")

    re4['Month'] = re4['Month'].map(months)
    re4['Day'] = re4['Day'].apply(lambda x: '0'+x if len(x) < 2 else x)
    re4['Year'] = re4['Year'].apply(lambda x: '19'+x if len(x) < 4 else x)
    re4 = re4[re4.Day.astype(int) < 32]
    re4 = re4[re4.Month.astype(int) < 13]

    df4 = pd.DataFrame((re4['Month']+'/'+re4['Day']+'/'+re4['Year']).astype('datetime64'), columns = ['Date'])
    df4.reset_index(inplace = True)
    df4.drop(['match'], axis = 1, inplace = True)
    df4 = df4.rename(columns={'level_0': 'Old_Index', 'Date': 'Date'})

    # Dates in Month yyyy, Mon yyyy, Mon, yyyy, Month, yyyy format    
    re5 = df.str.extractall(r'(?:(?P<Month>(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z./, ]*)(?P<Day>)(?P<Year>(?:19|20)\d{2}))') #set
    re5['Month'] = re5['Month'].str.replace(".", "").str.strip()
    re5['Month'] = re5['Month'].str.replace(",", "").str.strip()

    re5['Month'] = re5['Month'].map(months)
    re5['Day'] = re5['Day'].replace(np.nan, '01', regex = True)
    re5['Year'] = re5['Year'].apply(lambda x: '19'+x if len(x) < 4 else x)
    re5 = re5[re5.Month.astype(int) < 13]

    df5 = pd.DataFrame((re5['Month']+'/'+re5['Day']+'/'+re5['Year']).astype('datetime64'), columns = ['Date'])
    df5.reset_index(inplace = True)
    df5.drop(['match'], axis = 1, inplace = True)
    df5 = df5.rename(columns={'level_0': 'Old_Index', 'Date': 'Date'})

    # Dates in m/yyyy, mm/yyyy format    
    re6 = df.str.extractall(r'(?:(?P<Month>\d{1,2})[/](?P<Day>)(?P<Year>(?:19|20)?\d{4}))')

    re6['Month'] = re6['Month'].str.replace(".", "").str.strip()
    re6['Month'] = re6['Month'].str.replace(",", "").str.strip()

    re6['Month'] = re6['Month'].apply(lambda x: '0'+x if len(x) < 2 else x)
    re6['Day'] = re6['Day'].replace(np.nan, '01', regex = True)
    re6['Year'] = re6['Year'].apply(lambda x: '19'+x if len(x) < 4 else x)
    re6 = re6[re6.Day.astype(int) < 32]
    re6 = re6[re6.Month.astype(int) < 13]

    df6 = pd.DataFrame((re6['Month']+'/'+re6['Day']+'/'+re6['Year']).astype('datetime64'), columns = ['Date'])
    df6.reset_index(inplace = True)
    df6.drop(['match'], axis = 1, inplace = True)
    df6 = df6.rename(columns={'level_0': 'Old_Index', 'Date': 'Date'})

    # Dates in yyyy format
    re7 = df.str.extractall(r'(?:(?P<Month>)(?P<Day>)(?P<Year>(?:19|20)\d{2}))')

    re7['Month'] = re7['Month'].replace(np.nan, '01', regex = True)
    re7['Day'] = re7['Day'].replace(np.nan, '01', regex = True)
    re7['Year'] = re7['Year'].apply(lambda x: '19'+x if len(x) < 4 else x)

    df7 = pd.DataFrame((re7['Month']+'/'+re7['Day']+'/'+re7['Year']).astype('datetime64'), columns = ['Date'])
    df7.reset_index(inplace = True)
    df7.drop(['match'], axis = 1, inplace = True)
    df7 = df7.rename(columns={'level_0': 'Old_Index', 'Date': 'Date'})

    # Merge df1, df2, df3, df4 as they don't have overlap
    final_df = pd.concat([df1, df2, df3, df4])
    final_df = final_df.sort_values('Old_Index')
    final_df.reset_index(drop = True, inplace = True)

    # Merge df5 on 'outer' with final_df as it picks previous rows
    merge_df = pd.merge(final_df, df5, on = 'Old_Index', how = 'outer', validate = 'one_to_one')
    merge_df.Date_x.fillna(merge_df.Date_y, inplace = True)
    merge_df.drop(['Date_y'], axis = 1, inplace = True)
    merge_df.rename(columns = {'Date_x': 'Date'}, inplace = True)
    final_df = merge_df

    # Merge df6 on 'outer' similarly
    merge_df = pd.merge(final_df, df6, on = 'Old_Index', how = 'outer', validate = 'one_to_one')
    merge_df.Date_x.fillna(merge_df.Date_y, inplace = True)
    merge_df.drop(['Date_y'], axis = 1, inplace = True)
    merge_df.rename(columns = {'Date_x': 'Date'}, inplace = True)
    final_df = merge_df

    # Merge df7 similarly
    merge_df = pd.merge(final_df, df7, on = 'Old_Index', how = 'outer', validate = 'one_to_one')
    merge_df.Date_x.fillna(merge_df.Date_y, inplace = True)
    merge_df.drop(['Date_y'], axis = 1, inplace = True)
    merge_df.rename(columns = {'Date_x': 'Date'}, inplace = True)
    final_df = merge_df

    # Sort final_df
    final_df = final_df.sort_values('Date')

    # Convert Old_index to Series
    S1 = pd.Series(list(final_df['Old_Index']))

    return S1


# In[9]:


date_sorter()

