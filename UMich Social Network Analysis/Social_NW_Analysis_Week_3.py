
# coding: utf-8

# In[1]:


import networkx as nx

G = nx.karate_club_graph()


# In[2]:


G = nx.convert_node_labels_to_integers(G, first_label = 1)


# In[3]:


degCent = nx.degree_centrality(G)


# In[4]:


degCent[34] # node 34 has 17 connection and total 34 nodes, so 17/33


# In[5]:


degCent[33]


# In[6]:


G.is_directed()


# In[7]:


indegCent = nx.in_degree_centrality(G)


# In[8]:


outdegCent = nx.out_degree_centrality(G)


# In[11]:


closeCent = nx.closeness_centrality(G)


# In[13]:


closeCent[32]


# In[14]:


sum(nx.shortest_path_length(G,32).values())


# In[15]:


len(G.nodes())


# In[17]:


(len(G.nodes()) - 1)/(sum(nx.shortest_path_length(G,32).values()))


# In[18]:


Z = nx.Graph()
Z.add_edge('A', 'B')
Z.add_edge('A', 'C')
Z.add_edge('B', 'C')
Z.add_edge('C', 'D')
Z.add_edge('D', 'E')
Z.add_edge('E', 'F')
Z.add_edge('E', 'G')
Z.add_edge('F', 'G')


# In[23]:


nx.betweenness_centrality(Z, normalized = True, endpoints = False)


# In[24]:


import operator


# In[25]:


betCent = nx.betweenness_centrality(G, normalized = True, endpoints = False)

sorted(betCent.items(), key = operator.itemgetter(1), reverse = True)[0:5]


# In[28]:


betwCent_subset = nx.betweenness_centrality_subset(G, [34, 33, 21, 30, 16, 27, 15, 23, 10], [1, 4, 13, 11, 6, 12, 17, 7],
                                                   normalized = True)

sorted(betwCent_subset.items(), key = operator.itemgetter(1), reverse = True)[0:5]


# In[1]:


import networkx as nx


# In[2]:


Z = nx.Graph()
Z.add_edge('A', 'B')
Z.add_edge('A', 'C')
Z.add_edge('B', 'D')
Z.add_edge('C', 'D')
Z.add_edge('C', 'E')
Z.add_edge('D', 'E')
Z.add_edge('D', 'G')
Z.add_edge('E', 'G')
Z.add_edge('G', 'F')


# In[3]:


degCent = nx.degree_centrality(Z)


# In[4]:


degCent


# In[5]:


close = nx.closeness_centrality(Z)
close


# In[6]:


betn = nx.betweenness_centrality(Z)
betn


# In[11]:


nx.edge_betweenness_centrality(Z, normalized = False)


# In[12]:


P = nx.DiGraph()
P.add_edge('A', 'B')
P.add_edge('B', 'A')
P.add_edge('A', 'C')
P.add_edge('C', 'D')
P.add_edge('D', 'C')


# In[17]:


nx.pagerank(P, alpha = 0.5)


# In[18]:


Q = nx.DiGraph()
Q.add_edge('A', 'B')
Q.add_edge('A', 'C')
Q.add_edge('C', 'A')
Q.add_edge('C', 'D')
Q.add_edge('B', 'C')


# In[20]:


nx.pagerank(Q)


# In[24]:


nx.hits(Q)

