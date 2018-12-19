
# coding: utf-8

# In[1]:


import matplotlib.pyplot as plt
import mplleaflet
import pandas as pd

def leaflet_plot_stations(binsize, hashid):

    df = pd.read_csv('data/C2A2_data/BinSize_d{}.csv'.format(binsize))

    station_locations_by_hash = df[df['hash'] == hashid]

    lons = station_locations_by_hash['LONGITUDE'].tolist()
    lats = station_locations_by_hash['LATITUDE'].tolist()

    plt.figure(figsize=(8,8))

    plt.scatter(lons, lats, c='r', alpha=0.7, s=200)

    return mplleaflet.display()

leaflet_plot_stations(400,'fb441e62df2d58994928907a91895ec62c2c42e6cd075c2700843b89')


# In[ ]:


get_ipython().run_line_magic('matplotlib', 'notebook')


# In[ ]:


import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from datetime import datetime as dt
import matplotlib as mpl
mpl.get_backend()


# In[5]:


get_ipython().run_line_magic('matplotlib', 'notebook')

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from datetime import datetime as dt
import matplotlib as mpl

df = pd.read_csv('Weather.csv')

df['Date'] = list(map(pd.to_datetime, df['Date']))

df['MM_DD'] = df['Date'].dt.strftime('%m-%d')
df = df.set_index(df['MM_DD'])

df = df[~((df['MM_DD'] == '02-29'))]

df1 = df[df['Date'] < '2015-01-01']

Tmax = pd.DataFrame(df1.groupby('MM_DD').agg({'Data_Value': 'max', 'Date': 'first'}))
Tmax = Tmax.reset_index()
Tmax.columns = ['MM_DD', 'Temp_Max', 'Date']

Tmin = pd.DataFrame(df1.groupby('MM_DD').agg({'Data_Value': 'min', 'Date': 'first'}))
Tmin = Tmin.reset_index()
Tmin.columns = ['MM_DD', 'Temp_Min', 'Date']

plt.figure()
plt.plot(Tmax['MM_DD'], Tmax['Temp_Max'], '-', color = 'red', label = 'High Temps 2005-14')
plt.plot(Tmin['MM_DD'], Tmin['Temp_Min'], '-', color = 'blue', label = 'Low Temps 2005-14')

plt.gca().fill_between(Tmax['MM_DD'].values,
                       Tmax['Temp_Max'].values,
                       Tmin['Temp_Min'].values,
                       facecolor = 'khaki', alpha = 0.5)

df2 = df[df['Date'] >= '2015-01-01']

T2015max = pd.DataFrame(df2.groupby('MM_DD').agg({'Data_Value': 'max', 'Date': 'first'}))
T2015max = T2015max.reset_index()
T2015max.columns = ['MM_DD', 'Temp_Max', 'Date']

T2015min = pd.DataFrame(df2.groupby('MM_DD').agg({'Data_Value': 'min', 'Date': 'first'}))
T2015min = T2015min.reset_index()
T2015min.columns = ['MM_DD', 'Temp_Min', 'Date']

tempmax = []

for row1 in T2015max.itertuples():
    for row2 in Tmax.itertuples():
        if (row1.MM_DD == row2.MM_DD):           
            if (row1.Temp_Max > row2.Temp_Max):
                tempmax.append((row1.MM_DD, row1.Temp_Max, row1.Date))

df3 = pd.DataFrame.from_records(tempmax)
df3.columns = ['MM_DD', 'Temp_Max_All', 'Date']

tempmin = []

for row1 in T2015min.itertuples():
    for row2 in Tmin.itertuples():
        if (row1.MM_DD == row2.MM_DD):           
            if (row1.Temp_Min < row2.Temp_Min):
                tempmin.append((row1.MM_DD, row1.Temp_Min, row1.Date))

df4 = pd.DataFrame.from_records(tempmin)
df4.columns = ['MM_DD', 'Temp_Min_All', 'Date']

plt.scatter(df3['MM_DD'], df3['Temp_Max_All'], s = 10, color = 'indigo', label = 'Record High 2015')

plt.scatter(df4['MM_DD'], df4['Temp_Min_All'], s = 10, color = 'green', label = 'Record Low 2015')

plt.legend(loc = 'best', frameon = False)

plt.xticks(range(0, len(Tmax['MM_DD']),20), rotation = 45)

plt.gca().spines['right'].set_visible(False)

plt.gca().spines['top'].set_visible(False)

plt.xlabel('Day Of Year')

plt.ylabel('Temperature in Tenths of degrees C')

plt.title('Temperature Variations in Ann Arbor, MI 2005 - 2015')

plt.savefig('Temp_Vars.png')


# In[ ]:


df.head()


# In[ ]:


df['Date'] = list(map(pd.to_datetime, df['Date']))


# In[ ]:


df['MM_DD'] = df['Date'].dt.strftime('%m-%d')
df = df.set_index(df['MM_DD'])


# In[ ]:


df.head()


# In[ ]:


df = df[~((df['MM_DD'] == '02-29'))]


# In[ ]:


df.head()


# In[ ]:


df1 = df[df['Date'] < '2015-01-01']


# In[ ]:


df1['Daynum'] = df1['Date'].dt.dayofyear


# In[ ]:


Tmax = pd.DataFrame(df1.groupby('MM_DD').agg({'Data_Value': 'max', 'Date': 'first'}))
Tmax = Tmax.reset_index()
Tmax.columns = ['MM_DD', 'Temp_Max', 'Date']


# In[ ]:


Tmax.head()


# In[ ]:


Tmin = pd.DataFrame(df1.groupby('MM_DD').agg({'Data_Value': 'min', 'Date': 'first'}))
Tmin = Tmin.reset_index()
Tmin.columns = ['MM_DD', 'Temp_Min', 'Date']


# In[ ]:


Tmin.head()


# In[ ]:


plt.figure()
plt.plot(Tmax['MM_DD'], Tmax['Temp_Max'], '-', color = 'red', label = 'High Temps 2004-15')
plt.plot(Tmin['MM_DD'], Tmin['Temp_Min'], '-', color = 'blue', label = 'Low Temps 2004-15')

plt.gca().fill_between(Tmax['MM_DD'].values,
                       Tmax['Temp_Max'].values,
                       Tmin['Temp_Min'].values,
                       facecolor = 'khaki', alpha = 0.5)

df2 = df[df['Date'] >= '2015-01-01']

T2015max = pd.DataFrame(df2.groupby('MM_DD').agg({'Data_Value': 'max', 'Date': 'first'}))
T2015max = T2015max.reset_index()
T2015max.columns = ['MM_DD', 'Temp_Max', 'Date']

T2015min = pd.DataFrame(df2.groupby('MM_DD').agg({'Data_Value': 'min', 'Date': 'first'}))
T2015min = T2015min.reset_index()
T2015min.columns = ['MM_DD', 'Temp_Min', 'Date']

tempmax = []

for row1 in T2015max.itertuples():
    for row2 in Tmax.itertuples():
        if (row1.MM_DD == row2.MM_DD):           
            if (row1.Temp_Max > row2.Temp_Max):
                tempmax.append((row1.MM_DD, row1.Temp_Max, row1.Date))

df3 = pd.DataFrame.from_records(tempmax)
df3.columns = ['MM_DD', 'Temp_Max_All', 'Date']

tempmin = []

for row1 in T2015min.itertuples():
    for row2 in Tmin.itertuples():
        if (row1.MM_DD == row2.MM_DD):           
            if (row1.Temp_Min < row2.Temp_Min):
                tempmin.append((row1.MM_DD, row1.Temp_Min, row1.Date))

df4 = pd.DataFrame.from_records(tempmin)
df4.columns = ['MM_DD', 'Temp_Min_All', 'Date']

plt.scatter(df3['MM_DD'], df3['Temp_Max_All'], s = 10, color = 'indigo', label = 'Record High 2015')

plt.scatter(df4['MM_DD'], df4['Temp_Min_All'], s = 10, color = 'green', label = 'Record Low 2015')

plt.legend(loc = 'best', frameon = False, title = 'Legend')

plt.xticks(range(0, len(Tmax['MM_DD']),20), rotation = 45)

plt.gca().spines['right'].set_visible(False)

plt.gca().spines['top'].set_visible(False)

plt.xlabel('Day Of Year')

plt.ylabel('Temperature in Farenheit')

plt.title('Temperature Variations in Ann Arbor, MI 2004 - 2015')


# In[ ]:


df2 = df[df['Date'] >= '2015-01-01']

T2015max = pd.DataFrame(df2.groupby('MM_DD').agg({'Data_Value': 'max', 'Date': 'first'}))
T2015max = T2015max.reset_index()
T2015max.columns = ['MM_DD', 'Temp_Max', 'Date']

T2015min = pd.DataFrame(df2.groupby('MM_DD').agg({'Data_Value': 'min', 'Date': 'first'}))
T2015min = T2015min.reset_index()
T2015min.columns = ['MM_DD', 'Temp_Min', 'Date']

tempmax = []

for row1 in T2015max.itertuples():
    for row2 in Tmax.itertuples():
        if (row1.MM_DD == row2.MM_DD):           
            if (row1.Temp_Max > row2.Temp_Max):
                tempmax.append((row1.MM_DD, row1.Temp_Max, row1.Date))

df3 = pd.DataFrame.from_records(tempmax)
df3.columns = ['MM_DD', 'Temp_Max_All', 'Date']

tempmin = []

for row1 in T2015min.itertuples():
    for row2 in Tmin.itertuples():
        if (row1.MM_DD == row2.MM_DD):           
            if (row1.Temp_Min < row2.Temp_Min):
                tempmin.append((row1.MM_DD, row1.Temp_Min, row1.Date))

df4 = pd.DataFrame.from_records(tempmin)
df4.columns = ['MM_DD', 'Temp_Min_All', 'Date']

plt.scatter(df3['MM_DD'], df3['Temp_Max_All'], color = 'black')

plt.scatter(df4['MM_DD'], df4['Temp_Min_All'], color = 'yellow')


# In[ ]:


df2 = df[df['Date'] >= '2015-01-01']


# In[ ]:


df2.head()


# In[ ]:


T2015max = pd.DataFrame(df2.groupby('MM_DD').agg({'Data_Value': 'max', 'Date': 'first'}))
T2015max = T2015max.reset_index()
T2015max.columns = ['MM_DD', 'Temp_Max', 'Date']


# In[ ]:


T2015max.head()


# In[ ]:


T2015min = pd.DataFrame(df2.groupby('MM_DD').agg({'Data_Value': 'min', 'Date': 'first'}))
T2015min = T2015min.reset_index()
T2015min.columns = ['MM_DD', 'Temp_Min', 'Date']


# In[ ]:


T2015min.head()


# In[ ]:


tempmax = []

for row1 in T2015max.itertuples():
    for row2 in Tmax.itertuples():
        if (row1.MM_DD == row2.MM_DD):           
            if (row1.Temp_Max > row2.Temp_Max):
                tempmax.append((row1.MM_DD, row1.Temp_Max, row1.Date))

df3 = pd.DataFrame.from_records(tempmax)
df3.columns = ['MM_DD', 'Temp_Max_All', 'Date']
df3


# In[ ]:


tempmin = []

for row1 in T2015min.itertuples():
    for row2 in Tmin.itertuples():
        if (row1.MM_DD == row2.MM_DD):           
            if (row1.Temp_Min < row2.Temp_Min):
                tempmin.append((row1.MM_DD, row1.Temp_Min, row1.Date))

df4 = pd.DataFrame.from_records(tempmin)
df4.columns = ['MM_DD', 'Temp_Min_All', 'Date']
df4


# In[ ]:


plt.scatter(df3['MM_DD'], df3['Temp_Max_All'], color = 'black')


# In[ ]:


plt.scatter(df4['MM_DD'], df4['Temp_Min_All'], color = 'yellow')


# In[ ]:


T2015max = df[df['Date'].dt.strftime('%Y') == '2015']


# In[ ]:


T2015max.head()


# In[ ]:


T2015min = Tmin[Tmin['Date'].dt.strftime('%Y') == '2015']


# In[ ]:


T2015min.head()


# In[ ]:


# Tmax = pd.DataFrame(df.groupby(['MM-DD']).agg({'Data_Value': 'max', 'Date': 'First'}))
# # Tmax = Tmax.reset_index()
# # Tmax.columns = ['MM-DD', 'Temp_Max']

# Tmin = pd.DataFrame(df.groupby(['MM-DD'], as_index = False)['Data_Value'].min())
# # Tmin = Tmin.reset_index()
# Tmin.columns = ['MM-DD', 'Temp_Min']


# In[ ]:


Tmax


# In[ ]:


for index, row in Tmax.iterrows():
    for index1, row1 in T2015max.iterrows():
        if row['MM-DD'] == row1['MM-DD']:
            if row['Temp_Max'] < row1['Temp_Max']:
                print(row1['MM-DD'], row1['Temp_Max'])


# In[ ]:


for i in range(len(T2015max)):
    if row['MM-DD'] == row1['MM-DD']:
            if row['Temp_Max'] < row1['Temp_Max']:
                print(row1['MM-DD'], row1['Temp_Max'])


# In[ ]:


for index in range(len(Tmax)):
    for index1 in range(len(T2015max)):
        print(T2015max['MM-DD'])
        break


# In[ ]:


TScat


# In[ ]:


Tmax['MM-DD'] == T2015max['MM-DD']


# In[ ]:


for index, row in T2015max.iterrows():
    for index1, row1 in Tmax.iterrows():
        if row['MM-DD'] == row1['MM-DD']:
            print(row1['MM-DD'])
            print(row1['Temp_Max'])
#             print(row1['MM-DD']+' '+row1['Temp_Max'])
            if row['Temp_Max'] > row1['Temp_Max']:
                print(row['MM-DD'], row['Temp_Max'])


# In[ ]:


for index, row in T2015max.iterrows():
    if (row['MM-DD'] == Tmax['MM-DD']).any() and (row['Temp_Max'] > Tmax['Temp_Max']).any():
            print(row['MM-DD'])


# In[ ]:


T2015max['MM-DD']


# In[ ]:


Tmax.loc[10]['MM-DD']


# In[ ]:


Tmax['Temp_Max'].head()


# In[ ]:


T2015max['Temp_Max'].head()


# In[ ]:


TScat


# In[ ]:


df.Date.apply(lambda x: x[6:10])


# In[ ]:


df[df['MM-DD'] == '02-29']


# In[ ]:


import pandas as pd
import numpy as np
from pandas import DataFrame


df = pd.DataFrame({'Date': ['2015-05-08', '2015-05-07', '2015-05-06', '2015-05-05', '2015-05-08', '2015-05-07', '2015-05-06', '2015-05-05'], 'Sym': ['aapl', 'aapl', 'aapl', 'aapl', 'aaww', 'aaww', 'aaww', 'aaww'], 'Data2': [11, 8, 10, 15, 110, 60, 100, 40],'Data3': [5, 8, 6, 1, 50, 100, 60, 120]})


# In[ ]:


df


# In[ ]:


group = df['Data3'].groupby(df['Date']).sum()


# In[ ]:


group


# In[ ]:


df['Data4'] = group


# In[ ]:


df


# In[ ]:


df = pd.DataFrame({'Date': ['2015-05-08', '2015-05-07', '2015-05-06', '2015-05-05', '2015-05-08', '2015-05-07', '2015-05-06', '2015-05-05'], 'Sym': ['aapl', 'aapl', 'aapl', 'aapl', 'aaww', 'aaww', 'aaww', 'aaww'], 'Data2': [11, 8, 10, 15, 110, 60, 100, 40],'Data3': [5, 8, 6, 1, 50, 100, 60, 120]})


df


# In[ ]:


df['Data4'] = df['Data3'].groupby(df['Date']).transform('sum')


# In[ ]:


df


# In[ ]:


df1 = pd.DataFrame({"A": range(4), "B": ["PO", "PO", "PA", "PA"], "C": ["Est", "Est", "West", "West"]})
df1


# In[ ]:


df1.groupby('B').agg({'A':'max', 'C':'first'})

