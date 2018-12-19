
# coding: utf-8

# In[1]:


import nltk


# In[41]:


nltk.download('gutenberg')
nltk.download('genesis')
nltk.download('inaugural')
nltk.download('nps_chat')
nltk.download('webtext')
nltk.download('treebank')
nltk.download('udhr')
nltk.download('wordnet')
nltk.download('punkt')
nltk.download('tagsets')
nltk.download('averaged_perceptron_tagger')


# In[3]:


from nltk.book import *


# In[4]:


text1 #Loads Text1  Moby Dick into the


# In[5]:


sents() # Each sentence in text1


# In[6]:


sent1 # Just sentence 1


# In[7]:


text7 # Changing a corpus


# In[8]:


sent7 # Only sentence 7


# In[9]:


len(sent7) # No. of words in sent7


# In[10]:


len(text7) # No. of words in text7


# In[11]:


len(set(text7)) # Unique words in text7


# In[12]:


list(set(text7))[:10] # First 10 unique in text7; Might preceed with a u' which denotes the UTF-8 encoding


# In[13]:


dist = FreqDist(text7) # Unique words
len(dist)


# In[14]:


vocab1 = dist.keys() # First 10 unique in text7;
list(vocab1)[:10]


# In[15]:


dist['four'] # Times a particular word occurs


# In[16]:


freqwords = [w for w in vocab1 if len(w) > 5 and dist[w] > 100] # Freq words (Occuring atleast 100 times and len > 5)


# In[17]:


freqwords


# ## Normalization is transforming a word to make it appear the same way even though they look different

# In[18]:


input1 = "List listed lists listing listings"


# In[19]:


words = input1.lower().split(' ')


# In[20]:


words # All words are similar like they are lower case and split into individual words


# ## Stemming is to find the root word or root form of any word

# In[21]:


porter = nltk.PorterStemmer()


# In[22]:


[porter.stem(t) for t in words] # finds the root word for each entry in words


# ## Lemmatization is stemming but resulting stems are valid words

# In[23]:


udhr = nltk.corpus.udhr.words('English-Latin1')


# In[24]:


udhr[:20]


# In[25]:


[porter.stem(t) for t in udhr[:20]] # Stemming doesn't always work as univers and declar aren't exact words


# In[26]:


lemma = nltk.WordNetLemmatizer()


# In[27]:


[lemma.lemmatize(t) for t in udhr[:20]]


# ## Tokenization

# In[28]:


text11 = "Children shouldn't drink a sugary drink before bed."


# In[29]:


text11.split(' ')


# In[30]:


nltk.word_tokenize(text11)


# In[31]:


text12 = "This is the first sentence. A gallon of milk in the U.S. costs $2.99. Is this the third sentence? Yes, it is!"


# In[32]:


sentences = nltk.sent_tokenize(text12)


# In[33]:


len(sentences)


# In[34]:


sentences


# ### Parts Of Speech (POS) Tagging

# In[35]:


# POS are 
# nouns(NN),
# pronouns(PRP),
# adjectives(JJ), 
# verbs(VB), 
# cardinals(CD), 
# symbols(SYM),
# conjunctions(CC),
# determiner(DT),
# preposition(IN),
# modal(MD),
# possessive(POS),
# adverb(RB)
# ...


# In[38]:


nltk.help.upenn_tagset('MD')


# In[39]:


text13 = nltk.word_tokenize(text11)


# In[43]:


text13


# In[42]:


nltk.pos_tag(text13)


# In[44]:


text14 = nltk.word_tokenize("Visting aunts can be a nuisance")
nltk.pos_tag(text14)


# In[45]:


text15 = nltk.word_tokenize("Alice loves Bob")


# In[48]:


# Sentence has Noun Phrase (NP) and Verb Phrase (VP)
# VP can have Verb (V) and NP
grammar = nltk.CFG.fromstring(""" 
S -> NP VP
VP -> V NP
NP -> 'Alice'|'Bob'
V -> 'loves'""")


# In[49]:


parser = nltk.ChartParser(grammar)


# In[51]:


trees = parser.parse_all(text15)
for tree in trees:
    print(tree)


# In[55]:


text16 = nltk.word_tokenize('I saw the man with a telescope')
grammar1 = nltk.data.load('mygrammar.cfg')
grammar1


# In[56]:


parser = nltk.ChartParser(grammar1)
trees = parser.parse_all(text16)
for tree in trees:
    print(tree)


# In[58]:


from nltk.corpus import treebank
text17 = treebank.parsed_sents('wsj_0001.mrg')[0]
print(text17)


# In[59]:


text18 = nltk.word_tokenize('The old man the boat')
nltk.pos_tag(text18)


# In[61]:


text19 = nltk.word_tokenize('Colorless green ideas sleep furiously')
nltk.pos_tag(text19)


# In[62]:


nltk.word_tokenize('Zaina is a good girl ajhdl')

