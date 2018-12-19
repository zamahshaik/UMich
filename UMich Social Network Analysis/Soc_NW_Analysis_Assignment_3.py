
# coding: utf-8

# In[1]:


import networkx as nx
import operator

G1 = nx.read_gml('friendships.gml')


# In[2]:


# Question 1

# Find the degree centrality, closeness centrality, and normalized betweeness centrality (excluding endpoints) of node 100.

# This function should return a tuple of floats (degree_centrality, closeness_centrality, betweenness_centrality).

def answer_one():
        
    degCent = nx.degree_centrality(G1)[100]
    close = nx.closeness_centrality(G1)[100]
    betwn = nx.betweenness_centrality(G1, endpoints = False, normalized = True)[100]
    return (degCent, close, betwn)

# answer_one()


# In[3]:


# Question 2

# Suppose you are employed by an online shopping website and 
# are tasked with selecting one user in network G1 to send an 
# online shopping voucher to. We expect that the user who receives 
# the voucher will send it to their friends in the network. You want 
# the voucher to reach as many nodes as possible. The voucher can be 
# forwarded to multiple users at the same time, but the travel distance 
# of the voucher is limited to one step, which means if the voucher 
# travels more than one step in this network, it is no longer valid. 
# Apply your knowledge in network centrality to select the best candidate 
# for the voucher.

# This function should return an integer, the name of the node.

def answer_two():
        
    degCent = nx.degree_centrality(G1)
    degree = max(degCent.items(), key = operator.itemgetter(1))[0]
    
    return degree

# answer_two()


# In[4]:


# Question 3

# Now the limit of the voucher’s travel distance has been removed. 
# Because the network is connected, regardless of who you pick, every node in the network will 
# eventually receive the voucher. However, we now want to ensure that the voucher reaches the 
# nodes in the lowest average number of hops.

# How would you change your selection strategy? 
# Write a function to tell us who is the best candidate in the network under this condition.

# This function should return an integer, the name of the node.

def answer_three():
        
    closeCent = nx.closeness_centrality(G1)
    closen = max(closeCent.items(), key = operator.itemgetter(1))[0]
    
    return closen

# answer_three()


# In[5]:


# Question 4

# Assume the restriction on the voucher’s travel distance is still removed, 
# but now a competitor has developed a strategy to remove a person from the 
# network in order to disrupt the distribution of your company’s voucher. 
# Your competitor is specifically targeting people who are often bridges of 
# information flow between other pairs of people. Identify the single riskiest 
# person to be removed under your competitor’s strategy?

# This function should return an integer, the name of the node.

def answer_four():
        
    betCent = nx.betweenness_centrality(G1)
    betn = max(betCent.items(), key = operator.itemgetter(1))[0]
    
    return betn

# answer_four()


# In[6]:


# Part 2

# G2 is a directed network of political blogs, where nodes correspond to a blog and 
# edges correspond to links between blogs. Use your knowledge of PageRank and HITS to answer Questions 5-9.

G2 = nx.read_gml('blogs.gml')


# In[7]:


# Question 5

# Apply the Scaled Page Rank Algorithm to this network. 
# Find the Page Rank of node 'realclearpolitics.com' with damping value 0.85.

# This function should return a float.

def answer_five():
        
    scalPR = nx.pagerank(G2, alpha = 0.85)
        
    return scalPR['realclearpolitics.com']

# answer_five()


# In[43]:


# Question 6

# Apply the Scaled Page Rank Algorithm to this network with damping value 0.85. 
# Find the 5 nodes with highest Page Rank.

# This function should return a list of the top 5 blogs in desending order of Page Rank.

def answer_six():
        
    scalPR = nx.pagerank(G2, alpha = 0.85)
    scalPR5 = sorted(scalPR.keys(), key = lambda key:scalPR[key], reverse = True)[:5]
    
    return scalPR5

# answer_six()


# In[9]:


# Question 7

# Apply the HITS Algorithm to the network to find the hub and authority scores of node 'realclearpolitics.com'.

# Your result should return a tuple of floats (hub_score, authority_score).

def answer_seven():
        
    Hub_Score, Auth_Score = nx.hits(G2)
        
    return (Auth_Score['realclearpolitics.com'], Hub_Score['realclearpolitics.com'])

# answer_seven()


# In[42]:


# Question 8

# Apply the HITS Algorithm to this network to find the 5 nodes with highest hub scores.

# This function should return a list of the top 5 blogs in desending order of hub scores.

def answer_eight():
        
    hits = nx.hits(G2)
    hub5 = sorted(hits[0].keys(), key = lambda key: hits[0][key], reverse = True)[:5]
    
    return hub5

# answer_eight()


# In[41]:


# Question 9

# Apply the HITS Algorithm to this network to find the 5 nodes with highest authority scores.

# This function should return a list of the top 5 blogs in desending order of authority scores.

def answer_nine():
        
    hits = nx.hits(G2)
    auth5 = sorted(hits[1].keys(), key = lambda key:hits[1][key], reverse = True)[:5]
    
    return auth5

# answer_nine()

