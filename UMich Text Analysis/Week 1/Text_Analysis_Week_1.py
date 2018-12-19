
# coding: utf-8

# In[1]:


text1 = "Ethics are built right into the ideals and objectives of the United Nations "
len(text1)


# In[2]:


text2 = text1.split(' ')
text2


# In[3]:


len(text2)


# In[4]:


[w for w in text2 if len(w) > 3]


# In[5]:


[w for w in text2 if w.istitle()]


# In[6]:


[w for w in text2 if w.endswith('s')]


# In[7]:


text3 = 'To be or not to be'
text4 = text3.split(' ')

len(text4)


# In[8]:


set(text4)


# In[9]:


set([w.lower() for w in text4])


# In[10]:


# Word Comparisions

# s.startswith(t)
# s.endswith(t)
# t in s
# s.isupper(); s.islower(); s.istitle()
# s.isalpha(); -- alphabets or not
# s.isdigit(); -- number or not 0 - 9
# s.isalnum(); -- alphanumeric or not
# s.lower(); s.upper(); s.titlecase()
# s.split(t)
# s.splitlines() -- splits a sentence on new line character
# s.join(t)
# s.strip() -- removes all spaces from begin and end
# s.rstrip() -- removes white spaces on the right
# s.find(t) -- finds 't' from beginning
# s.rfind(t) -- finds 't' from end
# s.replace(u, v) -- replaces all 'u' with 'v'


# In[11]:


text5 = 'ouagadougou'
text6 = text5.split('ou')
text6


# In[12]:


'ou'.join(text6)


# In[13]:


list(text5)


# In[14]:


[c for c in text5]


# In[15]:


text7 = '  A quick brown fox jumped over the lazy dog. '
text7.split(' ')


# In[16]:


text7.strip().split()


# In[17]:


text7.find('o')


# In[18]:


text7.rfind('o')


# In[19]:


text7.replace('o', 'O')


# In[20]:


f = open('adspy_temp.dot', 'r')
texta = f.readline() # NOte the \n
texta


# In[21]:


texta.rstrip()


# In[22]:


f.seek(0)
text8 = f.read()
len(text8)


# In[23]:


text9 = text8.splitlines()
text9


# In[24]:


text9[3]


# In[25]:


# f = open(filename, mode) -- mode r or w
# f.readline(); f.read(); f.read(n) -- n chars
# for line in f: do something(line)
# f.seek()
# f.write(message)
# f.close()
# f.closed


# ## Regular Expressions

# In[26]:


text10 = '@UN @UN_Women "Ethics are built right into the ideals and objectives of the United Nations" #UNSG @ NY Society for Ethical Culture bit.ly/2guVelr'

text11 = text10.split(' ')
text11


# In[27]:


[w for w in text11 if w.startswith('#')]


# In[28]:


[w for w in text11 if w.startswith('@')] # extra @ appears


# In[29]:


import re


# In[30]:


[w for w in text11 if re.search('@[a-zA-Z0-9_]+', w)] # + denotes chars can occur multiple times


# In[ ]:


# . -- wildcard char, matches a single character
# ^ -- start of string
# $ -- end of string # $ comes after \n
# [] -- matches one of set of chars within the []
# [a-z] -- matches one of the chars between a to z
# [^abc] -- matches a char that is not a, b, c
# a|b -- matches with a or b where a and b are strings
# () -- scoping for operators
# \ -- escape character for special chars like (\t, \n, \b)
# \b -- matches word boundary
# \d -- any digit 0 - 9
# \D -- any non digit, equivalent to [^0-9]
# \s -- any whitespace, equivalent to [ \t\n\r\f\v]
# \S -- any non-whitespace, equivalent to [^ \t\n\r\f\v]
# \w -- Alphanumeric char, equivalent to [a-zA-Z0-9_]
# \W -- Non-alphanumeric char, equivalent to [^a-zA-z0-9_]

# Repetitions
# * -- matches 0 or more occurrences
# + -- matches 1 or more occurrences
# ? -- matches 0 or 1 occurrences
# {n} -- exactly n repetitions, n >= 0
# {n,} -- at least n repetitions
# {,n} -- at most n repetitions
# {m,n} -- at least m and at most n repetitions


# In[31]:


[w for w in text11 if re.search('@\w+', w)]


# In[33]:


text12 = 'ouagadougou'
re.findall(r'[aeiou]', text12)


# In[35]:


re.findall(r'[^aeiou]', text12)


# ## Date Handling

# In[36]:


date1 = '23-10-2002\n23/10/2002\n23/10/02\n10/23/2002\n23 Oct 2002\n23 October 2002\nOct 23, 2002\nOctober 23, 2002\n'


# In[38]:


re.findall(r'\d{2}[/-]\d{2}[/-]\d{4}', date1)


# In[40]:


re.findall(r'\d{2}[/-]\d{2}[/-]\d{2,4}', date1)


# In[42]:


re.findall(r'\d{1,2}[/-]\d{1,2}[/-]\d{2,4}', date1)


# In[43]:


re.findall(r'\d{2} (Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec) \d{4}', date1) 
#Returns only Oct instead of the entire date. () is a scoping operator and says find and return what's in ()


# In[44]:


re.findall(r'\d{2} (?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec) \d{4}', date1)


# In[45]:


re.findall(r'\d{2} (?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]* \d{4}', date1)


# In[48]:


re.findall(r'(?:\d{2} )?(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]* (?:\d{2}, )?\d{4}', date1)


# In[49]:


re.findall(r'(?:\d{1,2} )?(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]* (?:\d{1,2}, )?\d{2,4}', date1)


# ## Regex with Pandas

# In[50]:


import pandas as pd

time_sentences = ["Monday: The doctor's appointment is at 2:45pm.", 
                  "Tuesday: The dentist's appointment is at 11:30 am.",
                  "Wednesday: At 7:00pm, there is a basketball game!",
                  "Thursday: Be back home by 11:15 pm at the latest.",
                  "Friday: Take the train at 08:10 am, arrive at 09:00am."]

df = pd.DataFrame(time_sentences, columns = ['text'])
df


# In[53]:


df['text'].str.len()


# In[54]:


df['text'].str.split().str.len()


# In[55]:


df['text'].str.contains('appointment')


# In[56]:


df['text'].str.count(r'\d')


# In[57]:


df['text'].str.findall(r'\d')


# In[58]:


df['text'].str.findall(r'(\d?\d):(\d\d)')


# In[61]:


df['text']


# In[62]:


df['text'].str.replace(r'\w+day', '???')


# In[63]:


df['text'].str.replace(r'(\w+day\b)', lambda x: x.groups()[0][:3])


# In[64]:


df['text'].str.extract(r'(\d?\d):(\d\d)') # extract matches only first pattern; extractall matches all


# In[68]:


df['text'].str.extractall(r'((\d?\d):(\d\d) ?([ap]m))')


# In[71]:


df['text'].str.extractall(r'(?P<time>(?P<hour>\d?\d):(?P<minute>\d\d) ?(?P<period>[ap]m))')


# In[2]:


text13 = 'Qhamruzzamah Zamah Shaik Qhamru Zams Shaikh'
text14 = text13.split(' ')
text14


# In[5]:


import re


# In[13]:


[w for w in text14 if re.search('Qha\w+', w)]


# In[ ]:


re.findall(r'[^aeiou]', text12)


# In[3]:


text14.str.findall(r'Qham\w+')

