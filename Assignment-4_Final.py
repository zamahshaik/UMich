
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
from scipy.stats import ttest_ind
import re
from datetime import datetime


# In[2]:


def get_list_of_university_towns():
    Univdf = pd.DataFrame(columns = ['State', 'Towns', 'Univs'], index = range(567))
    f = open('university_towns.txt', 'r')
    line = f.readline()
    inx = 0
    state = ''
    town = ''
    univ = ''
    for inx in range(567):
        if '[edit]' in line:
            state = re.findall(r'(.*?)\[e',line)[0]
            line = f.readline()
        else:
            univ = re.findall(r'\((.*?)\)',line)
            town = re.findall(r'(\w+)\s\(', line)

            Univdf.loc[inx].State = state
            Univdf.loc[inx].Towns = town
            Univdf.loc[inx].Univs = univ
            line = f.readline()
    f.close()

    Univdf = Univdf.dropna()
    Univdf['RegionName'] = Univdf.Towns.apply(', '.join)
    Univdf['Univs'] = Univdf.Univs.apply(', '.join)
    Univtdf = pd.DataFrame(Univdf, columns = ['State', 'RegionName'])
    return Univtdf

get_list_of_university_towns()


# In[3]:


def get_recession_start():
    GDP_Qtr = pd.read_excel('gdplev.xls', skiprows = 5, parse_cols = [4, 5, 6], names = ['Quarter', 'GDP Current $', 'GDP 2009 $'])

    GDP_Qtr = GDP_Qtr.dropna()
    GDP_Qtr = GDP_Qtr[GDP_Qtr['Quarter'] >= '2000q1']
    GDP_Qtr = GDP_Qtr.reset_index(drop = True)
    
    q1 = 0 
    q2 = q1+1
    q3 = q2+1
    for index in range(len(GDP_Qtr) -1):
        if (GDP_Qtr.loc[q1]['GDP 2009 $'] > GDP_Qtr.loc[q2]['GDP 2009 $']):
            if (GDP_Qtr.loc[q2]['GDP 2009 $'] > GDP_Qtr.loc[q3]['GDP 2009 $']):
                RecPrev = GDP_Qtr.loc[q1]['Quarter']
                RecStart = GDP_Qtr.loc[q2]['Quarter']
                break
            q1 += 1
            q2 += 1
            q3 += 1
        else:
            q1 += 1
            q2 += 1
            q3 += 1
    return RecStart

get_recession_start()


# In[4]:


def get_recession_previous():
    GDP_Qtr = pd.read_excel('gdplev.xls', skiprows = 5, parse_cols = [4, 5, 6], names = ['Quarter', 'GDP Current $', 'GDP 2009 $'])

    GDP_Qtr = GDP_Qtr.dropna()
    GDP_Qtr = GDP_Qtr[GDP_Qtr['Quarter'] >= '2000q1']
    GDP_Qtr = GDP_Qtr.reset_index(drop = True)
    
    q1 = 0 
    q2 = q1+1
    q3 = q2+1
    for index in range(len(GDP_Qtr) -1):
        if (GDP_Qtr.loc[q1]['GDP 2009 $'] > GDP_Qtr.loc[q2]['GDP 2009 $']):
            if (GDP_Qtr.loc[q2]['GDP 2009 $'] > GDP_Qtr.loc[q3]['GDP 2009 $']):
                RecPrev = GDP_Qtr.loc[q1]['Quarter']
                break
            q1 += 1
            q2 += 1
            q3 += 1
        else:
            q1 += 1
            q2 += 1
            q3 += 1
    return RecPrev

get_recession_previous()


# In[5]:


def get_recession_end():
    GDP_Qtr = pd.read_excel('gdplev.xls', skiprows = 5, parse_cols = [4, 5, 6], names = ['Quarter', 'GDP Current $', 'GDP 2009 $'])

    GDP_Qtr = GDP_Qtr.dropna()
    GDP_Qtr = GDP_Qtr[GDP_Qtr['Quarter'] >= '2000q1']
    GDP_Qtr = GDP_Qtr.reset_index(drop = True)
    
    q1 = 0 
    q2 = q1+1
    q3 = q2+1
    q4 = q3+1
    q5 = q4+1
    Temp1 = ''
    S1 = []
    for index in range(len(GDP_Qtr)-1):
        if (GDP_Qtr.loc[q1]['GDP 2009 $'] > GDP_Qtr.loc[q2]['GDP 2009 $']):
            if (GDP_Qtr.loc[q2]['GDP 2009 $'] > GDP_Qtr.loc[q3]['GDP 2009 $']):
                if (GDP_Qtr.loc[q3]['GDP 2009 $'] < GDP_Qtr.loc[q4]['GDP 2009 $']):
                    if (GDP_Qtr.loc[q4]['GDP 2009 $'] < GDP_Qtr.loc[q5]['GDP 2009 $']):
                        RecEnd = GDP_Qtr.loc[q5]['Quarter']
            q1 += 1
            q2 += 1
            q3 += 1
            q4 += 1
            q5 += 1
        else:
            q1 += 1
            q2 += 1
            q3 += 1
            q4 += 1
            q5 += 1
    return RecEnd

get_recession_end()


# In[6]:


def get_recession_bottom():
    GDP_Qtr = pd.read_excel('gdplev.xls', skiprows = 5, parse_cols = [4, 5, 6], names = ['Quarter', 'GDP Current $', 'GDP 2009 $'])

    GDP_Qtr = GDP_Qtr.dropna()
    GDP_Qtr = GDP_Qtr[GDP_Qtr['Quarter'] >= '2000q1']
    GDP_Qtr = GDP_Qtr.reset_index(drop = True)

    q1 = 0 
    q2 = q1+1
    q3 = q2+1
    for index in range(len(GDP_Qtr) -1):
        if (GDP_Qtr.loc[q1]['GDP 2009 $'] > GDP_Qtr.loc[q2]['GDP 2009 $']):
            if (GDP_Qtr.loc[q2]['GDP 2009 $'] > GDP_Qtr.loc[q3]['GDP 2009 $']):
                RecStart = q2
                break
            #else:
                #return 1
            q1 += 1
            q2 += 1
            q3 += 1
        else:
            q1 += 1
            q2 += 1
            q3 += 1

    q1 = 0 
    q2 = q1+1
    q3 = q2+1
    q4 = q3+1
    q5 = q4+1
    Temp1 = ''
    S1 = []
    for index in range(len(GDP_Qtr)-1):
        if (GDP_Qtr.loc[q1]['GDP 2009 $'] > GDP_Qtr.loc[q2]['GDP 2009 $']):
            if (GDP_Qtr.loc[q2]['GDP 2009 $'] > GDP_Qtr.loc[q3]['GDP 2009 $']):
                if (GDP_Qtr.loc[q3]['GDP 2009 $'] < GDP_Qtr.loc[q4]['GDP 2009 $']):
                    if (GDP_Qtr.loc[q4]['GDP 2009 $'] < GDP_Qtr.loc[q5]['GDP 2009 $']):
                        RecEnd = q5
            q1 += 1
            q2 += 1
            q3 += 1
            q4 += 1
            q5 += 1
        else:
            q1 += 1
            q2 += 1
            q3 += 1
            q4 += 1
            q5 += 1

    GDP_Low = GDP_Qtr.loc[RecStart:RecEnd]
    RecBot = GDP_Low.loc[GDP_Low['GDP 2009 $'].idxmin()]['Quarter']
    return RecBot

get_recession_bottom()


# In[7]:


def convert_housing_data_to_quarters():
    Homesdf = pd.read_csv('City_Zhvi_AllHomes.csv')
    states = {'OH': 'Ohio', 'KY': 'Kentucky', 'AS': 'American Samoa', 'NV': 'Nevada', 'WY': 'Wyoming', 
              'NA': 'National', 'AL': 'Alabama', 'MD': 'Maryland', 'AK': 'Alaska', 'UT': 'Utah', 'OR': 'Oregon', 
              'MT': 'Montana', 'IL': 'Illinois', 'TN': 'Tennessee', 'DC': 'District of Columbia', 'VT': 'Vermont', 
              'ID': 'Idaho', 'AR': 'Arkansas', 'ME': 'Maine', 'WA': 'Washington', 'HI': 'Hawaii', 'WI': 'Wisconsin', 
              'MI': 'Michigan', 'IN': 'Indiana', 'NJ': 'New Jersey', 'AZ': 'Arizona', 'GU': 'Guam', 'MS': 'Mississippi', 
              'PR': 'Puerto Rico', 'NC': 'North Carolina', 'TX': 'Texas', 'SD': 'South Dakota', 
              'MP': 'Northern Mariana Islands', 
              'IA': 'Iowa', 'MO': 'Missouri', 'CT': 'Connecticut', 'WV': 'West Virginia', 'SC': 'South Carolina', 
              'LA': 'Louisiana', 'KS': 'Kansas', 'NY': 'New York', 'NE': 'Nebraska', 'OK': 'Oklahoma', 'FL': 'Florida', 
              'CA': 'California', 'CO': 'Colorado', 'PA': 'Pennsylvania', 'DE': 'Delaware', 'NM': 'New Mexico', 
              'RI': 'Rhode Island', 'MN': 'Minnesota', 'VI': 'Virgin Islands', 'NH': 'New Hampshire', 'MA': 'Massachusetts', 
              'GA': 'Georgia', 'ND': 'North Dakota', 'VA': 'Virginia'}
    Homesdf['State'].replace(states, inplace = True)    
    Homesdp = Homesdf.set_index(['State', 'RegionName'])
    cols = Homesdp.columns[0:49]
    Homesdp.drop(cols, axis = 1, inplace = True)
    Homesdp = (Homesdp.groupby(pd.PeriodIndex(Homesdp.columns, freq='Q'), axis=1)
                      .mean()
                      .rename(columns = lambda x: str(x).lower()))

    pd.options.display.float_format = '{:,.2f}'.format
    return Homesdp

convert_housing_data_to_quarters()


# In[8]:


def run_ttest():
    UnivTowns = get_list_of_university_towns()
    RecPrev = get_recession_previous()
    RecStart = get_recession_start()
    RecMid = get_recession_bottom()
    RecEnd = get_recession_end()
    Housing = convert_housing_data_to_quarters()

    Housing = Housing.reset_index()
    Housing['price_ratio'] = Housing[RecPrev]/Housing[RecMid]

    Houses_Univ_Towns = pd.merge(Housing, UnivTowns, how = 'inner', on = ['State', 'RegionName'])

    Houses_Univ_Towns = Houses_Univ_Towns.set_index(['State', 'RegionName']).dropna()

    Housing = Housing.set_index(['State', 'RegionName'])

    Houses_Non_Univ_Towns = pd.concat([Housing, Houses_Univ_Towns, Houses_Univ_Towns]).drop_duplicates(keep = False).dropna()

    t, p = ttest_ind(Houses_Univ_Towns['price_ratio'], Houses_Non_Univ_Towns['price_ratio'])

    different = True if p < 0.01 else False

    better = ("university town" 
              if Houses_Univ_Towns['price_ratio'].mean() < Houses_Non_Univ_Towns['price_ratio'].mean() 
              else "non-university town")

    return (different, p, better)



run_ttest()

