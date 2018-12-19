
# coding: utf-8

# In[1]:


import nltk
from nltk.corpus import wordnet as wn


# In[2]:


nltk.download('wordnet_ic')


# In[3]:


deer = wn.synset('deer.n.01')
elk = wn.synset('elk.n.01')
horse = wn.synset('horse.n.01')


# In[4]:


deer.path_similarity(elk)


# In[5]:


deer.path_similarity(horse)


# In[6]:


dog = wn.synset('dog.n.01')
puppy = wn.synset('puppy.n.01')


# In[7]:


from nltk.corpus import wordnet_ic
brown_ic = wordnet_ic.ic('ic-brown.dat')


# In[8]:


deer.lin_similarity(elk, brown_ic)


# In[9]:


deer.lin_similarity(horse, brown_ic)


# In[10]:


import nltk
from nltk.collocations import *


# In[11]:


bigram_measures = nltk.collocations.BigramAssocMeasures()

text = '''Nemegtomaia is estimated to have been around 2 m (7 ft) in length, 
and to have weighed 40 kg (85 lb). As an oviraptorosaur, 
it would have been feathered. It had a deep, narrow, and short skull, with an arched crest. 
It was toothless, had a short snout with a parrot-like beak, and a pair of tooth-like projections on its palate. 
It had three fingers; the first was largest and bore a strong claw. 
Nemegtomaia is classified as a member of the oviraptorid subfamily Ingeniinae, 
and it the only known member of this group with a cranial crest. 
Though Nemegtomaia has been used to suggest that oviraptorosaurs were flightless birds, 
the clade is generally considered a group of non-avian dinosaurs. '''

finder = BigramCollocationFinder.from_words(text)
finder.nbest(bigram_measures.pmi, 10) # top 10 pairs of bigram measures using the pmi method.
# finder.apply_freq_filter(2) # restricts to any bigram measure that occurs more than 10 times.


# In[12]:


import gensim
from gensim import corpora, models


# In[13]:


dictionary = corpora.Dictionary(doc_set)
corpus = [dictionary.doc2bow(doc) for doc in doc_set]
ldamodel = gensim.models.ldamodel.LdaModel(corpus, num_topics = 4, id2word = dictionary, passes = 50)
print(ldamodel.print_topics(num_topics = 4, num_words = 5)) # find 4 topics and return top 5 words

