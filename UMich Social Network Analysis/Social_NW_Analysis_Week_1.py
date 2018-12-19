
# coding: utf-8

# In[59]:


get_ipython().run_line_magic('matplotlib', 'notebook')
import networkx as nx
from networkx.algorithms import bipartite
import matplotlib.pyplot as plt

G1 = nx.Graph()
G1.add_edge('A', 'B')
G1.add_edge('B', 'C')


# In[4]:


G1.edges()


# In[5]:


G2 = nx.DiGraph()
G2.add_edge('B', 'A')
G2.add_edge('B', 'C')


# In[6]:


G2.edges()


# In[7]:


G3 = nx.Graph()
G3.add_edge('A', 'B', weight = 6)
G3.add_edge('B', 'C', weight = 13)


# In[8]:


G3.edges()


# In[14]:


G4 = nx.Graph()
G4.add_edge('A', 'B', sign = '+')
G4.add_edge('B', 'C', sign = '-')
G4.edges(data = True)


# In[15]:


G5 = nx.Graph()
G5.add_edge('A', 'B', relation = 'friend')
G5.add_edge('B', 'C', relation = 'family')
G5.add_edge('D', 'E', relation = 'neighbor')
G5.add_edge('E', 'I', relation = 'coworker')
G5.edges(data = 'relation')


# In[13]:


G6 = nx.MultiGraph()
G6.add_edge('A','B', relation = 'friend')
G6.add_edge('A','B', relation = 'neighbor')
G6.add_edge('G','F', relation = 'family')
G6.add_edge('G','F', relation = 'coworker')
G6


# In[18]:


G5['A']['B']


# In[19]:


G3['A']['B']['weight']


# In[21]:


G3['B']['A']['weight']


# In[23]:


G7 = nx.DiGraph()
G7.add_edge('A', 'B', weight = 6, relation = 'family')
G7.add_edge('B', 'C', weight = 13, relation = 'friend')

G7['B']['C']['weight']


# In[24]:


G7['C']['B']['weight'] ## error as relation C -> B isn't defined.


# In[25]:


G8 = nx.MultiGraph()
G8.add_edge('A', 'B', weight = 6, relation = 'family')
G8.add_edge('A', 'B', weight = 18, relation = 'friend')
G8.add_edge('C', 'B', weight = 13, relation = 'friend')

G8['A']['B']


# In[26]:


G8['A']['B'][0]['weight']


# In[28]:


G8 = nx.MultiDiGraph()
G8.add_edge('A', 'B', weight = 6, relation = 'family')
G8.add_edge('A', 'B', weight = 18, relation = 'friend')
G8.add_edge('C', 'B', weight = 13, relation = 'friend')

G8['B']['A'][0]['weight']


# In[29]:


G9 = nx.Graph()
G9.add_edge('A', 'B', weight = 6, relation = 'family')
G9.add_edge('B', 'C', weight = 13, relation = 'friend')

G9.add_node('A', role = 'trader')
G9.add_node('B', role = 'trader')
G9.add_node('C', role = 'manager')


# In[31]:


G9.edges()


# In[32]:


G9.nodes()


# In[33]:


G9.nodes(data = True)


# In[35]:


G9.nodes['A']['role']


# ## Bipartite Graphs

# In[37]:


B = nx.Graph()
B.add_nodes_from(['A', 'B', 'C', 'D', 'E'], bipartite = 0) # One set of nodes
B.add_nodes_from([1, 2, 3, 4], bipartite = 1)

B.add_edges_from([('A', 1), ('B', 1), ('C', 1), ('C', 3), ('D', 2), ('E', 3), ('E', 4)])


# In[38]:


bipartite.is_bipartite(B)


# In[39]:


B.add_edge('A', 'B')
bipartite.is_bipartite(B)


# In[40]:


B.remove_edge('A', 'B')


# In[41]:


X = set([1, 2, 3, 4])
bipartite.is_bipartite_node_set(B, X)


# In[46]:


X = set(['A', 'B', 'C', 'D', 'E'])
bipartite.is_bipartite_node_set(B, X)


# In[47]:


X = set([1, 2, 3, 4, 'A'])
bipartite.is_bipartite_node_set(B, X)


# In[48]:


bipartite.sets(B)


# In[75]:


B2 = nx.Graph()
B.add_edges_from([('A', 1), ('B', 1), ('C', 1), ('D', 1), ('H', 1), ('B', 2), ('C', 2), ('D', 2), ('E', 2), ('G', 2),
                  ('E', 3), ('F', 3), ('H', 3), ('J', 3), ('E', 4), ('I', 4), ('J', 4)])

X = set(['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J'])
P = bipartite.projected_graph(B, X)

nx.draw_networkx(P)


# In[81]:


B3 = nx.Graph()
B.add_edges_from([('A', 1), ('B', 1), ('C', 1), ('D', 1), ('H', 1), ('B', 2), ('C', 2), ('D', 2), ('E', 2), ('G', 2),
                  ('E', 3), ('F', 3), ('H', 3), ('J', 3), ('E', 4), ('I', 4), ('J', 4)])

X = set([1, 2, 3, 4])
P = bipartite.weighted_projected_graph(B, X)

nx.draw_networkx(P)


# In[86]:


import networkx as nx
from networkx.algorithms import bipartite

B = nx.Graph()
B.add_edges_from([('A', 'G'),('A','I'), ('B','H'), ('C', 'G'), ('C', 'I'),('D', 'H'), ('E', 'I'), ('F', 'G'), ('F', 'J')])
X1 = set(['A', 'B', 'C', 'D', 'E', 'F'])

P2 = bipartite.weighted_projected_graph(B, ['A', 'C'])

print(P2.nodes())

P2.edges(data = True)


# ### Loading Graphs in Networkx

# In[87]:


get_ipython().run_line_magic('matplotlib', 'notebook')
import networkx as nx
import numpy as np
import pandas as pd

G1 = nx.Graph()
G1.add_edges_from([(0, 1),
                   (0, 2),
                   (0, 3),
                   (0, 5),
                   (1, 3),
                   (1, 6),
                   (3, 4),
                   (4, 5),
                   (4, 7),
                   (5, 8),
                   (8, 9)])

nx.draw_networkx(G1)


# ### Adjacency List

# In[88]:


get_ipython().system('cat G_adjlist.txt')


# In[89]:


import networkx as nx

G=nx.MultiGraph()
G.add_node('A',role='manager')
G.add_edge('A','B',relation = 'friend')
G.add_edge('A','C', relation = 'business partner')
G.add_edge('A','B', relation = 'classmate')
G.node['A']['role'] = 'team member'
G.node['B']['role'] = 'engineer'


# In[92]:


G.nodes['A']['role']


# In[96]:


G['A']['B'][0]['relation']


# In[100]:


G['A']['C']


# In[102]:


G.nodes(data = True)

