
# coding: utf-8

# In[1]:


import pandas as pd
import re

pd.set_option('display.max_columns', None)

doc = []
with open('dates.txt') as file:
    for line in file:
        line = re.sub(",", "/", str(line))
        doc.append(line)
df = pd.Series(doc)
df.head(10)


# In[78]:


reg = df.str.findall(r'(?:\d{1,2}[/-]\d{1,2}[/-]?(?:19|20)?\d{2})|(?:(?:\d{2}[/ ])?(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec|January|February|March|April|May|June|July|August|September|October|November|December|Decemeber|Janaury)[./ ]* ?(?:19|20)?\d{2})|(?:19|20)\d{2}') #Full_Set


# In[79]:


reg.to_string()


# In[ ]:


[word.replace(",", "").replace(".", "") 
 for line in reg for word in line]


# In[ ]:


df.update(df.str.replace(",", " "))


# In[ ]:


df.to_string()


# In[ ]:


type(df)


# In[ ]:


str(x).replace(",", "/")


# In[84]:


s1 = doc[211]


# In[93]:


re.findall(r'(?:\d{1,2}[/-]\d{1,2}[/-]?(?:19|20)?\d{2})|(?:\d{2}[/ ])?(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec|January|February|March|April|May|June|July|August|September|October|November|December|Decemeber|Janaury)[. ](?:19|20)?\d{2}|(?:19|20)\d{2}', s1)


# In[85]:


s1


# In[ ]:


April 11, 1990


# In[41]:


s1


# In[ ]:


re.findall(r'')


# In[50]:


[int(match.group(-1)) for match in S1]


# In[48]:


S1


# In[14]:


s1


# In[18]:


s1 = df[213]


# In[ ]:


doc


# In[ ]:


reg = df.str.findall(r'(?:\d{1,2}[/-]\d{1,2}[/-]?(?:19|20)?\d{2})|(?:\d{2} )?(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec|January|February|March|April|May|June|July|August|September|October|November|December|Decemeber|Janaury)[., ]* ?(?:19|20)?\d{2}|(?:19|20)\d{2}') #Full_Set


# In[ ]:


reg.to_string()


# In[ ]:


regc = re.compile('(?:\d{1,2}[/-]\d{1,2}[/-]?(?:19|20)?\d{2})|(?:\d{2} )?(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec|January|February|March|April|May|June|July|August|September|October|November|December|Decemeber|Janaury)[., ]* ?(?:19|20)?\d{2}|(?:19|20)\d{2}')

regz = regc.sub(lambda m:m.group().replace(',', ''))

print(regz)


# In[ ]:


reg = [[x.str.sub(r",","/") for x in l] for l in reg]


# In[ ]:


for l in reg:
    for x in l:
        x = str(x).replace(",", "/")


# In[ ]:


reg = [[x.replace(r",","/") for x in l] for l in reg]


# In[ ]:


reg.to_string()


# In[ ]:


months = {'January': 1, 'February': 2, 'March': 3, 'April': 4, 'May': 5, 'June': 6, 
              'July': 7, 'August': 8, 'September': 9, 'October': 10, 'November': 11, 'December': 12, 
              'Decemeber': 12, 'Janaury': 1,
              'Jan': 1, 'Feb': 2, 'Mar': 3, 'Apr': 4, 'May': 5, 'Jun': 6, 
              'Jul': 7, 'Aug': 8, 'Sep': 9, 'Oct': 10, 'Nov': 11, 'Dec': 12}

months['January']


# In[ ]:


import string

mess = 'Sample message! Notice: It has Punctuation'


# In[ ]:


string.punctuation


# In[ ]:


example = [["string 1", "a\r\ntest string:"],["string 1", "test 2: another\r\ntest string"]]
example = [[x.replace('\r\n','') for x in l] for l in example]
print(example)


# In[ ]:


nopunc = [char for char in mess if char not in string.punctuation]
nopunc = ''.join(nopunc)


# In[ ]:


nopunc = ''.join(nopunc)


# In[ ]:


nopunc


# In[ ]:


l = [[0, 4], [2, 3], [5, 2]]
dict = ({0: 'da-1.txt',
 1: 'da-2.txt',
 2: 'en-1.txt',
 3: 'en-2.txt',
 4: 'it-1.txt',
 5: 'it-2.txt'})

l


# In[ ]:


l = [[0, 4], [2, 3], [5, 2]]
dict = ({0: 'da-1.txt',
 1: 'da-2.txt',
 2: 'en-1.txt',
 3: 'en-2.txt',
 4: 'it-1.txt',
 5: 'it-2.txt'})

def rec_replace(l, d):
    for i in range(len(l)):
        if isinstance(l[i], list):
            rec_replace(l[i], d)
        else :
            l[i] = d.get(l[i], l[i])

rec_replace(l, dict)


# In[ ]:


l


# In[ ]:


def rec_replace(l, d):
    for i in range(len(l)):
        if isinstance(l[i], list):
            rec_replace(l[i], d)
        else :
            l[i] = d.get(l[i], l[i])

rec_replace(re, months)


# In[ ]:


re.to_string()


# In[ ]:


flat_list = [item for sublist in reg for item in sublist]
re1 = pd.DataFrame(flat_list, columns = ['Dates'])


# In[ ]:


re1[182:212]


# In[ ]:


[x for x in re if (x.replace(r',', '') or y.str.replace(r',', ''))]


# In[ ]:


(lambda x: '01/01/' + x if len(x) == 4 else x)


# In[ ]:


list =[ ['a','b'], ['a','c'], ['b','d'] ]
Search = 'c'

# return if it find in either item 0 or item 1
print([x for x,y in list if x == Search or y == Search])

# return if it find in item 1
print([x for x,y in list if x == Search])


# In[ ]:


re1['Dates'] = re1['Dates'].apply(lambda x: '01/01/' + x if len(x) == 4 else x)


# In[ ]:


for i in re1.iloc['Dates']:
    print((re1.iloc['Dates'][i]))


# In[ ]:


re1['Dates'] = re1['Dates'].apply(lambda x: '01/' + x if x == (r'(\d{2}\\d{4})') else x)


# In[ ]:


re1.to_string()


# In[ ]:


re1.loc[re1['Dates']]


# In[ ]:


len('05/2015')


# In[ ]:


# re1 = df1['Data'].str.findall(r'(?:\d{1,2}[/-]\d{1,2}[/-]\d{2,4})') #set

# re1 = df1['Data'].str.findall(r'(?:\d{1,2}[/-]\d{1,2}[/-]?(?:19|20)?\d{2})') #set

re1 = df.str.findall(r'(?:\d{1,2}[/-]\d{1,2}[/-]?(?:19|20)?\d{2})|(?:19|20)\d{2}') #Set

re2 = df.str.findall(r'(?:\d{2} )?(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z., ]* (?:\d{2}, )?(?:\d{2} )?\d{4}') #set


# In[ ]:


re1 = df.str.findall(r'(?:\d{1,2}[/-]\d{1,2}[/-]?(?:19|20)?\d{2})|(?:19|20)\d{2}')


# In[ ]:


re2 = df.str.findall(r'(?:\d{2} )?(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z., ]* (?:\d{2}, )?(?:\d{2} )?\d{4}') #set


# In[ ]:


import string

df1 = pd.DataFrame()

for list in re2:
    for item in list:
#         Date = ''.join(item)
        re2[item] = str(item).replace(",", "/")

#         print(Date)
#         str(item).replace("\,", " ")
#         print(item)
#         D1 = ' '.join(map(str, item))
#         D1 = "[{0}]".format(" ".join(str(item)))
#         print(D1)
#         print(("".join(str(x) for x in item)))
        
# print(int("".join(str(x) for x in [7,7,7,7])))       
        
#         print(item)
#         for ele in enumerate(item):
#             print(ele)
#         for i in range(len(item))
#         no_punc = [char for char in item if char not in string.punctuation]
#         no_punc = ''.join(no_punc)
#         re2[index] = no_punc

            
# nopunc = [char for char in mess if char not in string.punctuation]
# nopunc = ''.join(nopunc)            


# In[ ]:


re2.to_string()


# In[ ]:


re2 = df.str.findall(r'(?:\d{2} )?(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z. ]*[^,] (?:\d{2}, )?(?:\d{2} )?\d{4}') #set


# In[ ]:


re2


# In[ ]:


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


# In[ ]:


dates_extracted.to_string()


# In[ ]:


re2.to_string()


# In[ ]:


pd.set_option('display.max_columns', None) 


# In[ ]:


df.to_string()


# In[ ]:


print(int("".join(str(x) for x in [7,7,7,7])))


# In[ ]:


s = 'Oct 20 2011'
for word in s.split():
    print(word)


# In[ ]:


for i in re:
    if len(re[i]) < 5:
        re[i].append('01/01/')

re.to_string()


# In[ ]:


df1 = pd.DataFrame(df, columns = ['Data'])


# In[ ]:


df1.head()


# In[ ]:


re = df.str.findall(r'(?:\d{1,2}[/-]\d{1,2}[/-]?(?:19|20)?\d{2})|(?:\d{2} )?(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec|January|February|March|April|May|June|July|August|September|October|November|December|Decemeber|Janaury)[., ]* ?(?:19|20)?\d{2}|(?:19|20)\d{2}') #Full_Set


# In[ ]:


re.to_string()


# In[ ]:


Date = '02-10-1981'

print(type(Date))

re.findall(r'\d\d[-](0[1-9]|1[012])[-](0[1-9]|[12][0-9]|3[01])', Date)


# In[ ]:


S1 = pd.Series(['Zamah', 'Qhamruz', 'Qhamruzzamah'], name = 'Namees')
S1


# In[ ]:


df_dates = df.str.extract(r'((?:\d{1,2})?[-/\s,]{0,2}(?:\d{1,2})?(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec|January|February|March|April|May|June|July|August|September|October|November|December)?[-/\s,]{0,2}(?:19|20)?\d{2})')


# In[ ]:


df_dates.to_string()


# In[ ]:


doc

