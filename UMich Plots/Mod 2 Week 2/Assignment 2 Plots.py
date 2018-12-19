
# coding: utf-8

# In[1]:


import matplotlib.pyplot as plt
import mplleaflet
import pandas as pd
import numpy as np
from datetime import datetime as dt
import matplotlib as mpl
mpl.get_backend()

get_ipython().run_line_magic('matplotlib', 'notebook')

def leaflet_plot_stations(binsize, hashid):

    df = pd.read_csv('data/C2A2_data/BinSize_d{}.csv'.format(binsize))

    station_locations_by_hash = df[df['hash'] == hashid]

    lons = station_locations_by_hash['LONGITUDE'].tolist()
    lats = station_locations_by_hash['LATITUDE'].tolist()

    plt.figure(figsize=(8,8))

    plt.scatter(lons, lats, c='r', alpha=0.7, s=200)

    return mplleaflet.display()

leaflet_plot_stations(400,'fb441e62df2d58994928907a91895ec62c2c42e6cd075c2700843b89')


# In[2]:


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

