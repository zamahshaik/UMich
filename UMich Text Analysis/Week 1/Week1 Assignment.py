
# coding: utf-8

# In[43]:


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


# In[17]:


re4 = df.str.extractall(r'(?:(?P<Month>(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z. ]*)(?P<Day>\d{2} )(?P<Year>(?:19|20)?\d{2}))') #set


# In[19]:


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


# In[95]:


re5 = df.str.extractall(r'(?:(?P<Month>(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z./, ]*)(?P<Day>)(?P<Year>(?:19|20)\d{2}))') #set


# In[96]:


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


# In[195]:


re6 = df.str.extractall(r'(?:(?P<Month>\d{1,2})[/](?P<Year>(?:19|20)?\d{2}))')


# In[196]:


re6


# In[ ]:


re7 = df.str.extractall(r'(?:((?:19|20)\d{2}))')


# In[ ]:


re7


# # Merge DataFrames

# In[177]:


final_df = pd.concat([df1, df2, df3, df4])


# In[178]:


final_df = final_df.sort_values('Old_Index')


# In[179]:


final_df.reset_index(drop = True, inplace = True)


# In[180]:


final_df.head()


# In[181]:


merge_df = pd.merge(final_df, df5, on = 'Old_Index', how = 'outer', validate = 'one_to_one')


# In[182]:


merge_df.Date_x.fillna(merge_df.Date_y, inplace = True)


# In[183]:


merge_df.drop(['Date_y'], axis = 1, inplace = True)


# In[191]:


merge_df.rename(columns = {'Date_x': 'Date'}, inplace = True)


# In[193]:


final_df = merge_df


# In[194]:


final_df


# In[ ]:


re3 = df.str.findall(r'(?:19|20)\d{2}')


# In[ ]:


# flat_list = [item for sublist in re1 for item in sublist]
# df1 = pd.DataFrame(flat_list, columns = ['Dates'])


# In[ ]:


# final_df['Date'] = pd.to_datetime(final_df['Date'])

final_df.sort_values('Date')

# final_df


# In[ ]:


final_df2.to_string()


# In[93]:


re5.loc[228]

