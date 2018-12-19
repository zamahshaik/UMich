
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import re

pd.set_option('display.max_columns', None)

doc = []
with open('dates.txt') as file:
    for line in file:
#         line = re.sub(",", "/", str(line))
        doc.append(line)
df = pd.Series(doc)
df.head(10)


# In[2]:


re1 = df.str.extractall(r'(?:(?P<Month>\d{1,2})[/-](?P<Day>\d{1,2})[/-](?P<Year>(?:19|20)?\d{2}))')


# In[3]:


re1['Month'] = re1['Month'].apply(lambda x: '0'+x if len(x) < 2 else x)
re1['Day'] = re1['Day'].apply(lambda x: '0'+x if len(x) < 2 else x)
re1['Year'] = re1['Year'].apply(lambda x: '19'+x if len(x) < 4 else x)
re1 = re1[re1.Day.astype(int) < 32]
re1 = re1[re1.Month.astype(int) < 13]

df1 = pd.DataFrame((re1['Month']+'/'+re1['Day']+'/'+re1['Year']).astype('datetime64'), columns = ['Date'])
df1.reset_index(inplace = True)
df1.drop(['match'], axis = 1, inplace = True)
df1 = df1.rename(columns={'level_0': 'Old_Index', 'Date': 'Date'})

df1.head()


# In[4]:


re2 = df.str.extractall(r'(?:(?P<Day>\d{2} )(?P<Month>(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z.,/ ]*) (?P<Year>(?:19|20)?\d{2}))') #set


# In[5]:


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

df2.head()


# In[6]:


re3 = df.str.extractall(r'(?:(?P<Month>(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z. ]*)(?P<Day>\d{2}[/., ] )(?P<Year>(?:19|20)?\d{2}))') #set


# In[7]:


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

df3.head()


# In[8]:


re4 = df.str.extractall(r'(?:(?P<Month>(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z. ]*)(?P<Day>\d{2} )(?P<Year>(?:19|20)?\d{2}))') #set


# In[9]:


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

df4.head()


# In[10]:


re5 = df.str.extractall(r'(?:(?P<Month>(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z./, ]*)(?P<Day>)(?P<Year>(?:19|20)\d{2}))') #set


# In[11]:


re5['Month'] = re5['Month'].str.replace(".", "").str.strip()
re5['Month'] = re5['Month'].str.replace(",", "").str.strip()

re5['Month'] = re5['Month'].map(months)
re5['Day'] = re5['Day'].replace(np.nan, '01', regex = True)
re5['Year'] = re5['Year'].apply(lambda x: '19'+x if len(x) < 4 else x)
re5 = re5[re5.Day.astype(int) < 32]
re5 = re5[re5.Month.astype(int) < 13]

df5 = pd.DataFrame((re5['Month']+'/'+re5['Day']+'/'+re5['Year']).astype('datetime64'), columns = ['Date'])
df5.reset_index(inplace = True)
df5.drop(['match'], axis = 1, inplace = True)
df5 = df5.rename(columns={'level_0': 'Old_Index', 'Date': 'Date'})

df5.head()


# In[12]:


re6 = df.str.extractall(r'(?:(?P<Month>\d{1,2})[/](?P<Day>)(?P<Year>(?:19|20)?\d{4}))')


# In[13]:


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

df6.head()


# In[14]:


re7 = df.str.extractall(r'(?:(?P<Month>)(?P<Day>)(?P<Year>(?:19|20)\d{2}))')


# In[15]:


re7['Month'] = re7['Month'].replace(np.nan, '01', regex = True)
re7['Day'] = re7['Day'].replace(np.nan, '01', regex = True)
re7['Year'] = re7['Year'].apply(lambda x: '19'+x if len(x) < 4 else x)

df7 = pd.DataFrame((re7['Month']+'/'+re7['Day']+'/'+re7['Year']).astype('datetime64'), columns = ['Date'])
df7.reset_index(inplace = True)
df7.drop(['match'], axis = 1, inplace = True)
df7 = df7.rename(columns={'level_0': 'Old_Index', 'Date': 'Date'})

df7.head()


# # Merge DataFrames

# In[16]:


final_df = pd.concat([df1, df2, df3, df4])
final_df = final_df.sort_values('Old_Index')
final_df.reset_index(drop = True, inplace = True)
final_df.head()


# In[17]:


merge_df = pd.merge(final_df, df5, on = 'Old_Index', how = 'outer', validate = 'one_to_one')
merge_df.Date_x.fillna(merge_df.Date_y, inplace = True)
merge_df.drop(['Date_y'], axis = 1, inplace = True)
merge_df.rename(columns = {'Date_x': 'Date'}, inplace = True)
final_df = merge_df
final_df.head()


# In[18]:


merge_df = pd.merge(final_df, df6, on = 'Old_Index', how = 'outer', validate = 'one_to_one')
merge_df.Date_x.fillna(merge_df.Date_y, inplace = True)
merge_df.drop(['Date_y'], axis = 1, inplace = True)
merge_df.rename(columns = {'Date_x': 'Date'}, inplace = True)
final_df = merge_df
final_df.head()


# In[19]:


merge_df = pd.merge(final_df, df7, on = 'Old_Index', how = 'outer', validate = 'one_to_one')
merge_df.Date_x.fillna(merge_df.Date_y, inplace = True)
merge_df.drop(['Date_y'], axis = 1, inplace = True)
merge_df.rename(columns = {'Date_x': 'Date'}, inplace = True)
final_df = merge_df
final_df.head()


# In[20]:


final_df = final_df.sort_values('Date')


# In[21]:


S1 = pd.Series(list(final_df['Old_Index']))


# In[22]:


S1


# In[23]:


# flat_list = [item for sublist in re1 for item in sublist]
# df1 = pd.DataFrame(flat_list, columns = ['Dates'])
# merge_df.query('Date_x == "NaT"')


# In[24]:


# final_df['Date'] = pd.to_datetime(final_df['Date'])

final_df.sort_values('Date')

# final_df


# In[25]:


re4


# In[26]:




# Your code here
# Full date
global df
dates_extracted = df.str.extractall(r'(?P<origin>(?P<month>\d?\d)[/|-](?P<day>\d?\d)[/|-](?P<year>\d{4}))')
index_left = ~df.index.isin([x[0] for x in dates_extracted.index])
dates_extracted = dates_extracted.append(df[index_left].str.extractall(r'(?P<origin>(?P<month>\d?\d)[/|-](?P<day>([0-2]?[0-9])|([3][01]))[/|-](?P<year>\d{2}))'))
index_left = ~df.index.isin([x[0] for x in dates_extracted.index])
del dates_extracted[3]
del dates_extracted[4]
dates_extracted = dates_extracted.append(df[index_left].str.extractall(r'(?P<origin>(?P<day>\d?\d) ?(?P<month>[a-zA-Z]{3,})\.?,? (?P<year>\d{4}))'))
index_left = ~df.index.isin([x[0] for x in dates_extracted.index])
dates_extracted = dates_extracted.append(df[index_left].str.extractall(r'(?P<origin>(?P<month>[a-zA-Z]{3,})\.?-? ?(?P<day>\d\d?)(th|nd|st)?,?-? ?(?P<year>\d{4}))'))
del dates_extracted[3]
index_left = ~df.index.isin([x[0] for x in dates_extracted.index])

# Without day
dates_without_day = df[index_left].str.extractall('(?P<origin>(?P<month>[A-Z][a-z]{2,}),?\.? (?P<year>\d{4}))')
dates_without_day = dates_without_day.append(df[index_left].str.extractall(r'(?P<origin>(?P<month>\d\d?)/(?P<year>\d{4}))'))
dates_without_day['day'] = 1
dates_extracted = dates_extracted.append(dates_without_day)
index_left = ~df.index.isin([x[0] for x in dates_extracted.index])

# Only year
dates_only_year = df[index_left].str.extractall(r'(?P<origin>(?P<year>\d{4}))')
dates_only_year['day'] = 1
dates_only_year['month'] = 1
dates_extracted = dates_extracted.append(dates_only_year)
index_left = ~df.index.isin([x[0] for x in dates_extracted.index])

# Year
dates_extracted['year'] = dates_extracted['year'].apply(lambda x: '19' + x if len(x) == 2 else x)
dates_extracted['year'] = dates_extracted['year'].apply(lambda x: str(x))

# Month
dates_extracted['month'] = dates_extracted['month'].apply(lambda x: x[1:] if type(x) is str and x.startswith('0') else x)
month_dict = dict({'September': 9, 'Mar': 3, 'November': 11, 'Jul': 7, 'January': 1, 'December': 12,
                   'Feb': 2, 'May': 5, 'Aug': 8, 'Jun': 6, 'Sep': 9, 'Oct': 10, 'June': 6, 'March': 3,
                   'February': 2, 'Dec': 12, 'Apr': 4, 'Jan': 1, 'Janaury': 1,'August': 8, 'October': 10,
                   'July': 7, 'Since': 1, 'Nov': 11, 'April': 4, 'Decemeber': 12, 'Age': 8})
dates_extracted.replace({"month": month_dict}, inplace=True)
dates_extracted['month'] = dates_extracted['month'].apply(lambda x: str(x))

# Day
dates_extracted['day'] = dates_extracted['day'].apply(lambda x: str(x))

# Cleaned date
dates_extracted['date'] = dates_extracted['month'] + '/' + dates_extracted['day'] + '/' + dates_extracted['year']
dates_extracted['date'] = pd.to_datetime(dates_extracted['date'])

dates_extracted.sort_values(by='date', inplace=True)
df1 = pd.Series(list(dates_extracted.index.labels[0]))

