
# coding: utf-8

# In[1]:


import networkx as nx


# In[2]:


G = nx.karate_club_graph()
G = nx.convert_node_labels_to_integers(G, first_label = 1)


# In[3]:


nx.average_shortest_path_length(G)


# In[4]:


nx.radius(G)


# In[5]:


nx.diameter(G)


# In[6]:


nx.center(G)


# In[7]:


nx.periphery(G)


# In[8]:


nx.eccentricity(G)


# In[9]:


import networkx as nx 

G = nx.DiGraph()
G.add_edges_from([('A', 'B'), ('C', 'A'), ('A', 'E'), ('G', 'A'), ('A', 'N'), ('B', 'C'), ('D', 'B'), ('B', 'E'), ('C', 'D'), ('E', 'C'), ('D', 'E'), ('E', 'D'), ('F', 'G'), ('I', 'F'), ('J', 'F'), ('H', 'G'), ('I', 'G'), ('G', 'J'), ('I', 'H'), ('H', 'I'), ('I', 'J'), ('J', 'O'), ('O', 'J'), ('K', 'M'), ('K', 'L'), ('O', 'K'), ('O', 'L'), ('N', 'L'), ('L', 'M'), ('N', 'O')])


# In[10]:


nx.minimum_edge_cut(G, 'H', 'O')


# In[21]:


get_ipython().run_line_magic('matplotlib', 'notebook')

import networkx as nx
import matplotlib.pyplot as plt

G = nx.read_gpickle('major_us_cities')


# In[23]:


G.nodes(data = True)


# In[16]:


nx.__version__

