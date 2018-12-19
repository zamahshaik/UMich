
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
from scipy.stats import ttest_ind
import re


# In[2]:


f = open('university_towns.txt', 'r')
content = f.read()
f.close()

print(content)


# In[9]:


f1 = open('university_towns.txt')
while True:
    line = f1.readline()
    if line.find('[edit]') == -1:
        print(line)
    else:
        line = f1.readline()
f1.close()


# In[10]:


z1 = pd.read_fwf('university_towns.txt')
z1


# In[3]:


States = []
Towns = []
Univs = []


# In[7]:


re.findall(r'(\w+)*\[e', content) #re.findall(r'[a-zA-Z0-9]*--[a-zA-Z0-9]*', book)


# In[129]:


State = re.findall(r'\n(.*?)\[e',content, re.M)
State


# In[4]:


Univs = re.findall(r'\((.*?)\)',content,re.M)


# In[5]:


Univs


# In[57]:


Towns = re.findall(r'(\w+)\s\(', content, re.M)


# In[58]:


Towns


# In[ ]:


Ser1 = pd.DataFrame(Towns, Univs, columns=['Towns', 'Univs'])


# In[ ]:


r'([\w\.-]+)@


# In[ ]:


def search(text,n):
    '''Searches for text, and retrieves n words either side of the text, which are retuned seperatly'''
    word = r"\W*([\w]+)"
    groups = re.search(r'{}\W*{}{}'.format(word*n,'place',word*n), text).groups()
    return groups[:n],groups[n:]

search()


# In[60]:


def search(text,n):
    '''Searches for text, and retrieves n words either side of the text, which are retuned seperatly'''
    groups = re.findall(r'[a-zA-Z]*[edit]', text)
    print(groups)
    return groups[n]

search(content, 20)


# In[26]:


re.search(r'Co+kie', 'Cooooooooooookieooooo').group()


# In[27]:


re.search(r'Co*kie', 'Cooooooooooookieooooo').group()


# In[29]:


re.search(r'Cok?ie', 'Cooooooooooookieooooo').group()


# In[30]:


re.search(r'Colou?r', 'Color').group()


# In[33]:


email_address = 'Please contact us at: support@datacamp.com'
match = re.search(r'([\w\.-]+)@([\w\.-]+)', email_address)
if match:
  print(match.group()) # The whole matched text
  print(match.group(1)) # The username (group 1)
  print(match.group(2)) # The host (group 2)


# In[44]:


email_address = "Please contact us at: support-center@datacamp.com, xyz@datacamp.com"

#'addresses' is a list that stores all the possible match
addresses = re.findall(r'[\w\.-]+@[\w\.-]+', email_address)
print(addresses)
for address in addresses: 
    print(address)


# In[20]:


xyz = 'Alabama [edit]'
word = r"\W*([\w]+)"
re.findall(r'{}([\W+])'.format(word*1, '[edit]'), xyz)


# In[24]:


if xyz.endswith('[edit]'):
    print(xyz[0:])


# In[27]:


line = "Cats are smarter than dogs"

matchObj = re.match( r'(.*) are (.*?) .*', line, re.M|re.I)

if matchObj:
   print( "matchObj.group() : ", matchObj.group())
   print( "matchObj.group(1) : ", matchObj.group(1))
   print( "matchObj.group(2) : ", matchObj.group(2))
else:
   print( "No match!!")


# In[38]:


xyz = 'Alabama [edit]'
serc = re.match(r'[edit]', xyz)
print(serc)


# In[92]:


txt="I like to eat apple. Me too. Let's go buy some apples."
txt = "." + txt
print(txt)
re.findall(r"\."+".+"+"apple"+".+"+"\.", txt)


# In[3]:


cols = ['c1', 'c2', 'c3']
df2 = pd.DataFrame(columns=cols, index=range(2))
for a in range(2):
    df2.loc[a].c1 = 4
    df2.loc[a].c2 = 5
    df2.loc[a].c3 = 6
df2


# In[ ]:


cols = Homesdp.columns[0:49]
Homesdp.drop(cols, axis = 1, inplace = True)
Homesdp


# In[ ]:


year_ctr = Homesdp.columns[0][0:4]
q1s = set(['-01', '-02', '-03'])
q1s


# In[ ]:


year_ctr = Homesdp.columns[0][0:4]
q1s = set(['01', '02', '03'])
Q1sum = 0.00
for i in Homesdp.columns:
    if Homesdp.columns[0][0:4] == year_ctr:
#         if Homesdp.columns[0][5:7] in q1s:
        Homesdp['Q1sum'] = Q1sum + Homesdp[i].values

Homesdp

Homesdp[Homesdp.columns[0][0:4]+'-01']


# In[ ]:


Z1 = Homesdp[Homesdp.columns[0:3]]
Z1

Z1['q1'] = Z1.fillna(0).rolling(3, axis = 1).mean()
Z1

pd.to_datetime(['2000-04', '2000-05', '2000-06']).to_period('Q')


# In[ ]:


import pandas as pd

# make a simple dataframe
df = pd.DataFrame({'a':[1,2], 'b':[3,4]})
df
#    a  b
# 0  1  3
# 1  2  4

# create an unattached column with an index
df.apply(lambda row: row.a + row.b, axis=1)
# 0    4
# 1    6

# do same but attach it to the dataframe
df['c'] = df.apply(lambda row: row.a + row.b, axis=1)
df
#    a  b  c
# 0  1  3  4
# 1  2  4  6

