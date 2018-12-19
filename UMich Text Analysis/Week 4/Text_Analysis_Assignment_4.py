
# coding: utf-8

# In[1]:


import numpy as np
import nltk
from nltk.corpus import wordnet as wn
import pandas as pd


def convert_tag(tag):
    """Convert the tag given by nltk.pos_tag to the tag used by wordnet.synsets"""
    
    tag_dict = {'N': 'n', 'J': 'a', 'R': 'r', 'V': 'v'}
    try:
        return tag_dict[tag[0]]
    except KeyError:
        return None

def doc_to_synsets(doc):
    """
    Returns a list of synsets in document.

    Tokenizes and tags the words in the document doc.
    Then finds the first synset for each word/tag combination.
    If a synset is not found for that combination it is skipped.

    Args:
        doc: string to be converted

    Returns:
        list of synsets

    Example:
        doc_to_synsets('Fish are nvqjp friends.')
        Out: [Synset('fish.n.01'), Synset('be.v.01'), Synset('friend.n.01')]
    """
    tokens = nltk.word_tokenize(doc)
    postags = nltk.pos_tag(tokens)
    tags = [tag[1] for tag in postags]
    contag = [convert_tag(tag) for tag in tags]
    listag = list(zip(tokens, contag))
    sysets = [wn.synsets(x, y) for x, y in listag]
    result = [val[0] for val in sysets if len(val) > 0]
    
    return result

def similarity_score(s1, s2):
    """
    Calculate the normalized similarity score of s1 onto s2

    For each synset in s1, finds the synset in s2 with the largest similarity value.
    Sum of all of the largest similarity values and normalize this value by dividing it by the
    number of largest similarity values found.

    Args:
        s1, s2: list of synsets from doc_to_synsets

    Returns:
        normalized similarity score of s1 onto s2

    Example:
        synsets1 = doc_to_synsets('I like cats')
        synsets2 = doc_to_synsets('I like dogs')
        similarity_score(synsets1, synsets2)
        Out: 0.73333333333333339
    """
    p = []
    for i in s1:
        q = []
        for j in s2:
            q.append(i.path_similarity(j))
        result = [x for x in q if x is not None]
        if len(result) > 0:
            p.append(max(result))
    return sum(p)/len(p)

def document_path_similarity(doc1, doc2):
    """Finds the symmetrical similarity between doc1 and doc2"""

    synsets1 = doc_to_synsets(doc1)
    synsets2 = doc_to_synsets(doc2)

    return (similarity_score(synsets1, synsets2) + similarity_score(synsets2, synsets1)) / 2


# In[2]:


def test_document_path_similarity():
    doc1 = 'This is a function to test document_path_similarity.'
    doc2 = 'Use this function to see if your code in doc_to_synsets     and similarity_score is correct!'
    return document_path_similarity(doc1, doc2)
# test_document_path_similarity()


# In[3]:


# Use this dataframe for questions most_similar_docs and label_accuracy
paraphrases = pd.read_csv('paraphrases.csv')
paraphrases.head()


# In[4]:


def most_similar_docs():
    
    paraphrases['sim_score'] = paraphrases.apply(lambda x: document_path_similarity(x['D1'], x['D2']), axis = 1)
    paraphrases.sort_values(by = 'sim_score', ascending = False, inplace = True)
    sim_docs = (paraphrases.iloc[0]['D1'], paraphrases.iloc[0]['D2'], paraphrases.iloc[0]['sim_score'])    
    return sim_docs

# most_similar_docs()


# In[5]:


def label_accuracy():
    from sklearn.metrics import accuracy_score
    
    paraphrases['sim_score'] = paraphrases.apply(lambda x: document_path_similarity(x['D1'], x['D2']), axis = 1)
    paraphrases['pred'] = np.where(paraphrases['sim_score'] > 0.75, 1, 0)
    
    acc_score = accuracy_score(paraphrases['Quality'], paraphrases['pred'])
    return acc_score

# label_accuracy()


# In[6]:


# Part 2 - Topic Modelling

# For the second part of this assignment, you will use Gensim's LDA (Latent Dirichlet Allocation) model 
# to model topics in newsgroup_data. 
# You will first need to finish the code in the cell below by using gensim.models.ldamodel.LdaModel constructor 
# to estimate LDA model parameters on the corpus, and save to the variable ldamodel. 
# Extract 10 topics using corpus and id_map, and with passes=25 and random_state=34.


# In[7]:


import pickle
import gensim
from sklearn.feature_extraction.text import CountVectorizer

# Load the list of documents
with open('newsgroups', 'rb') as f:
    newsgroup_data = pickle.load(f)

# Use CountVectorizor to find three letter tokens, remove stop_words, 
# remove tokens that don't appear in at least 20 documents,
# remove tokens that appear in more than 20% of the documents
vect = CountVectorizer(min_df=20, max_df=0.2, stop_words='english', 
                       token_pattern='(?u)\\b\\w\\w\\w+\\b')
# Fit and transform
X = vect.fit_transform(newsgroup_data)

# Convert sparse matrix to gensim corpus.
corpus = gensim.matutils.Sparse2Corpus(X, documents_columns=False)

# Mapping from word IDs to words (To be used in LdaModel's id2word parameter)
id_map = dict((v, k) for k, v in vect.vocabulary_.items())


# In[8]:


# Use the gensim.models.ldamodel.LdaModel constructor to estimate 
# LDA model parameters on the corpus, and save to the variable `ldamodel`

# Your code here:
ldamodel = gensim.models.ldamodel.LdaModel(corpus = corpus, id2word = id_map, passes = 25, random_state = 34)


# In[9]:


# lda_topics

# Using ldamodel, find a list of the 10 topics and the most significant 10 words in each topic. 
# This should be structured as a list of 10 tuples where each tuple takes on the form:

# (9, '0.068*"space" + 0.036*"nasa" + 0.021*"science" + 0.020*"edu" + 0.019*"data" + 0.017*"shuttle" + 
#  0.015*"launch" + 0.015*"available" + 0.014*"center" + 0.014*"sci"')

# for example.

# This function should return a list of tuples.

def lda_topics():
    
    return ldamodel.print_topics(num_topics = 10, num_words = 10)

# lda_topics()


# In[10]:


# topic_distribution

# For the new document new_doc, find the topic distribution. 
# Remember to use vect.transform on the the new doc, and Sparse2Corpus to convert the sparse matrix to gensim corpus.

# This function should return a list of tuples, where each tuple is (#topic, probability)

new_doc = ["\n\nIt's my understanding that the freezing will start to occur because of the\ngrowing distance of Pluto and Charon from the Sun, due to it's\nelliptical orbit. It is not due to shadowing effects. \n\n\nPluto can shadow Charon, and vice-versa.\n\nGeorge Krumins\n-- "]

def topic_distribution():
    
    import gensim

    sparmat = vect.transform(new_doc)

    corpus = gensim.matutils.Sparse2Corpus(sparmat, documents_columns=False)

    ldamodel = gensim.models.ldamodel.LdaModel(corpus = corpus, num_topics = 10, random_state = 35, passes = 25)

    topics = ldamodel.get_document_topics(corpus)

    topic_dist = []
    for val in list(topics):
        for v in val:
            topic_dist.append(v)
    
    return topic_dist

# topic_distribution()


# In[11]:


# topic_names

# From the list of the following given topics, assign topic names to the topics you found. 
# If none of these names best matches the topics you found, create a new 1-3 word "title" for the topic.

# Topics: Health, Science, Automobiles, Politics, Government, Travel, Computers & IT, Sports, Business, 
#     Society & Lifestyle, Religion, Education.

# This function should return a list of 10 strings.

def topic_names():
    
    topics = ['Science', 'Religion', 'Education', 'Health', 'Automobiles', 'Politics', 'Government', 'Travel', 'Computers & IT', 
          'Sports']
    
    return topics

# topic_names()

