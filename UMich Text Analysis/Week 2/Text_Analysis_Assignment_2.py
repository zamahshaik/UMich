
# coding: utf-8

# In[1]:


import nltk
import pandas as pd
import numpy as np

# If you would like to work with the raw text you can use 'moby_raw'
with open('moby.txt', 'r') as f:
    moby_raw = f.read()
    
# If you would like to work with the novel in nltk.Text format you can use 'text1'
moby_tokens = nltk.word_tokenize(moby_raw)
text1 = nltk.Text(moby_tokens)


# In[2]:


def example_one():
    
    return len(nltk.word_tokenize(moby_raw)) # or alternatively len(text1)

example_one()


# In[3]:


def example_two():
    
    return len(set(nltk.word_tokenize(moby_raw))) # or alternatively len(set(text1))

example_two()


# In[4]:


from nltk.stem import WordNetLemmatizer

def example_three():

    lemmatizer = WordNetLemmatizer()
    lemmatized = [lemmatizer.lemmatize(w,'v') for w in text1]

    return len(set(lemmatized))

example_three()


# In[5]:


# Question 1

# What is the lexical diversity of the given text input? (i.e. ratio of unique tokens to the total number of tokens)

# This function should return a float.


def answer_one():
    
    total = len(text1)
    unique = len(set(text1))
    
    lexdiv = unique/total
    
    return lexdiv

# answer_one()


# In[6]:


# Question 2

# What percentage of tokens is 'whale'or 'Whale'?

# This function should return a float.
from nltk import FreqDist

def answer_two():
    
    dist = FreqDist(text1)
    word1 = 'whale'
    word2 = 'Whale'
    whaper = ((dist[word1]+dist[word2])*100/len(text1))
    
    return whaper

# answer_two()


# In[7]:


# Question 3

# What are the 20 most frequently occurring (unique) tokens in the text? What is their frequency?

# This function should return a list of 20 tuples where each tuple is of the form (token, frequency). 
# The list should be sorted in descending order of frequency.

def answer_three():
    
    dist = FreqDist(text1)
    freq20 = dist.most_common(20)
    
    return freq20

# answer_three()


# In[8]:


# Question 4

# What tokens have a length of greater than 5 and frequency of more than 150?

# This function should return a sorted list of the tokens that match the above constraints. To sort your list, use sorted()

def answer_four():
    freqdist = FreqDist(text1)

    vocab1 = freqdist.keys()

    freqwords = sorted([w for w in vocab1 if len(w) > 5 and freqdist[w] > 150])
    
    return freqwords

# answer_four()


# In[9]:


# Question 5

# Find the longest word in text1 and that word's length.

# This function should return a tuple (longest_word, length).

def answer_five():
    maxlen = max(len(word) for word in text1)
    longest = sorted([(w, len(w)) for w in text1 if len(w) == maxlen])[0]
        
    return longest

# answer_five()


# In[10]:


# Question 6

# What unique words have a frequency of more than 2000? What is their frequency?

# "Hint: you may want to use isalpha() to check if the token is a word and not punctuation."

# This function should return a list of tuples of the form (frequency, word) sorted in descending order of frequency.

def answer_six():
    
    freqdist = FreqDist(text1)

    vocab1 = freqdist.keys()

    freq2000 = sorted([(freqdist[w], w) for w in vocab1 if freqdist[w] > 2000 and w.isalpha()], reverse = True)
    
    return freq2000

# answer_six()


# In[11]:


# Question 7

# What is the average number of tokens per sentence?

# This function should return a float.


def answer_seven():
    
    avgtok = np.mean([len(nltk.word_tokenize(sents)) for sents in nltk.sent_tokenize(moby_raw)])
    return avgtok

# answer_seven()


# In[12]:


# Question 8

# What are the 5 most frequent parts of speech in this text? What is their frequency?

# This function should return a list of tuples of the form (part_of_speech, frequency) 
# sorted in descending order of frequency.

def answer_eight():
    
    z = nltk.pos_tag(text1)
    tags = nltk.FreqDist(tag for (word, tag) in z)

    return tags.most_common(5)

# answer_eight()


# In[13]:


from nltk.corpus import words

correct_spellings = words.words()


# In[14]:


# Question 9

# For this recommender, your function should provide recommendations for the three default words provided 
# above using the following distance metric:

# Jaccard distance on the trigrams of the two words.

# This function should return a list of length three: 
# ['cormulent_reccomendation', 'incendenece_reccomendation', 'validrate_reccomendation'].

def answer_nine(entries=['cormulent', 'incendenece', 'validrate']):
    
    correction = []
    for i in entries:
        corr_word = [word for word in correct_spellings if word.startswith(i[0])]
        jacc_dist = [(nltk.jaccard_distance(set(nltk.ngrams(i, n = 3)), set(nltk.ngrams(word, n = 3))), word) for word in corr_word]
        correction.append(sorted(jacc_dist)[0][1])
    
    return correction
    
# answer_nine()


# In[15]:


# Question 10

# For this recommender, your function should provide recommendations for the three default words provided 
# above using the following distance metric:

# Jaccard distance on the 4-grams of the two words.

# This function should return a list of length three: ['cormulent_reccomendation', 'incendenece_reccomendation', 'validrate_reccomendation'].


def answer_ten(entries=['cormulent', 'incendenece', 'validrate']):
    correction = []
    for i in entries:
        corr_word = [word for word in correct_spellings if word.startswith(i[0])]
        jacc_dist = [(nltk.jaccard_distance(set(nltk.ngrams(i, n = 4)), set(nltk.ngrams(word, n = 4))), word) for word in corr_word]
        correction.append(sorted(jacc_dist)[0][1])
    
    return correction
    
# answer_ten()


# In[16]:


# Question 11

# For this recommender, your function should provide recommendations for the three default words provided 
# above using the following distance metric:

# Edit distance on the two words with transpositions.

# This function should return a list of length three: ['cormulent_reccomendation', 'incendenece_reccomendation', 'validrate_reccomendation'].

def answer_eleven(entries=['cormulent', 'incendenece', 'validrate']):
    correction = []
    for i in entries:
        corr_word = [word for word in correct_spellings if word.startswith(i[0])]
        edit_dist = [(nltk.edit_distance(i, word), word) for word in corr_word]
        correction.append(sorted(edit_dist)[0][1])
    
    return correction
    
# answer_eleven()

