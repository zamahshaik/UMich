
# coding: utf-8

# # Assignment 2
# 
# Before working on this assignment please read these instructions fully. In the submission area, you will notice that you can click the link to **Preview the Grading** for each step of the assignment. This is the criteria that will be used for peer grading. Please familiarize yourself with the criteria before beginning the assignment.
# 
# An NOAA dataset has been stored in the file `data/C2A2_data/BinnedCsvs_d200/258b90b9e8b4bf81821195315b1b12705641fb5b57c0533e601363ec.csv`. The data for this assignment comes from a subset of The National Centers for Environmental Information (NCEI) [Daily Global Historical Climatology Network](https://www1.ncdc.noaa.gov/pub/data/ghcn/daily/readme.txt) (GHCN-Daily). The GHCN-Daily is comprised of daily climate records from thousands of land surface stations across the globe.
# 
# Each row in the assignment datafile corresponds to a single observation.
# 
# The following variables are provided to you:
# 
# * **id** : station identification code
# * **date** : date in YYYY-MM-DD format (e.g. 2012-01-24 = January 24, 2012)
# * **element** : indicator of element type
#     * TMAX : Maximum temperature (tenths of degrees C)
#     * TMIN : Minimum temperature (tenths of degrees C)
# * **value** : data value for element (tenths of degrees C)
# 
# For this assignment, you must:
# 
# 1. Read the documentation and familiarize yourself with the dataset, then write some python code which returns a line graph of the record high and record low temperatures by day of the year over the period 2005-2014. The area between the record high and record low temperatures for each day should be shaded.
# 2. Overlay a scatter of the 2015 data for any points (highs and lows) for which the ten year record (2005-2014) record high or record low was broken in 2015.
# 3. Watch out for leap days (i.e. February 29th), it is reasonable to remove these points from the dataset for the purpose of this visualization.
# 4. Make the visual nice! Leverage principles from the first module in this course when developing your solution. Consider issues such as legends, labels, and chart junk.
# 
# The data you have been given is near **Jūrmala, Jurmala, Latvia**, and the stations the data comes from are shown on the map below.

# In[113]:


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

leaflet_plot_stations(200,'258b90b9e8b4bf81821195315b1b12705641fb5b57c0533e601363ec')


# In[114]:


import pandas as pd
import numpy as np
import csv
import matplotlib as mpl
import datetime 
import matplotlib.pyplot as plt
import matplotlib.dates as mdates


# In[115]:


df = pd.read_csv('data/C2A2_data/BinnedCsvs_d200/258b90b9e8b4bf81821195315b1b12705641fb5b57c0533e601363ec.csv')
#df = pd.read_csv('City_Zhvi_AllHomes.csv', skiprows=0,)


# In[116]:


df.sort(columns = ['Date','Element','Data_Value']).head()


# In[117]:


Year = df['Date'].str[:4]


# In[118]:


MD = df['Date'].str[5:]


# In[119]:


MD.head()


# In[120]:


df['Year'] = Year


# In[121]:


df['MD'] = MD


# In[122]:


df = df[df['MD'] != '02-29']
#for i in range(len(df['Data_Value'])):
#    df['Data_Value'].loc[i] = df['Data_Value'].loc[i]/10


# In[123]:


df['Data_Value'] = np.divide(df['Data_Value'],10)


# In[124]:


df1 = df[df['Year'] != '2015']
df2 = df[df['Year'] == '2015']


# In[125]:


f1 = {'Data_Value':['min']}
f2 = {'Data_Value':['max']}    


# In[126]:


MinData = df1.groupby('MD').agg(f1)
MaxData = df1.groupby('MD').agg(f2)


# In[127]:


MinData15 = df2.groupby('MD').agg(f1)
MaxData15 = df2.groupby('MD').agg(f2)


# In[128]:


record_min = np.where(MinData['Data_Value'] > MinData15['Data_Value'])
record_max = np.where(MaxData['Data_Value'] < MaxData15['Data_Value'])
record_min = record_min[0]
record_max = record_max[0]
MinData15.iloc[record_min].head()


# In[129]:


MaxData = df1.groupby('MD').agg(f2)
X = np.array(MinData.values)
Y = np.array(MaxData.values)


# In[130]:


get_ipython().run_line_magic('matplotlib', 'inline')

plt.figure(figsize=(15,10))
ax = plt.gca()
plt.plot(list(X),label = 'Min temp 2005-2014', c = '#9994f2', alpha=0.7 )
plt.plot(list(Y),label = 'Max temp 2005-2014', c = '#cc6a6a', alpha=0.7)

plt.xlabel('Month of the year', fontsize=20)
plt.ylabel('Temperature (degrees Celsius)', fontsize=20)
plt.title('Temperature in Latvia in 2005-2015', fontsize=25)

plt.scatter(record_min, MinData15.iloc[record_min], label = 'Record low 2015',s = 10 )
plt.scatter(record_max, MaxData15.iloc[record_max], label = 'Record high 2015',c = 'r',s = 10)

plt.gca().fill_between(range(len(X)),X[:, 0], Y[:, 0],
                 facecolor='grey', alpha=0.2)

plt.axhline(y=0, xmin=0.046, xmax=0.954,c='grey',ls = '--')

#def test(axes):
    #axes.bar(x,y)
#    axes.set_xticks(x)
#    axes.set_xticklabels([i+100 for i in x])

plt.legend(fontsize=15)

#x1 = plt.gca().xaxis

#for item in x1.get_ticklabels():
#    item.set_rotation(45)

import datetime as dt

m = ['Jan','Feb','Mar','Apr','May','June','July','Aug','Sept','Oct','Nov','Dec']
# This is the vital step. It will create a list of day numbers corresponding to middle of each month i.e. 15(Jan), 46(Feb), ... 
ticks = [(dt.date(2017,m,1)-dt.date(2016,12,15)).days for m in range(1,13)]
# It is important to use a non-leap year for this calculation (I used 2017).
# Also, I used (2016,12,15) to substract so that I get middle of each month rather than beginning, it just looks better that way.



ax.set_xticks(ticks)
ax.set_xticklabels(m,fontsize=15)   

