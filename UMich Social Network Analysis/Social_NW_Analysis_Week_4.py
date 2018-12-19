
# coding: utf-8

# In[ ]:


Dict = {'Onion': 23, 'Bread': 40, 'Milk': 20}


# In[ ]:


Dict


# In[ ]:


sorted(Dict.values(), reverse = True)


# In[1]:


get_ipython().run_line_magic('matplotlib', 'notebook')
import networkx as nx
import matplotlib.pyplot as plt


# In[ ]:


G = nx.Graph()
G.add_edge('A', 'B')
G.add_edge('A', 'G')
G.add_edge('A', 'H')
G.add_edge('B', 'C')
G.add_edge('C', 'D')
G.add_edge('C', 'E')
G.add_edge('D', 'F')
G.add_edge('F', 'G')
G.add_edge('F', 'I')
G.add_edge('G', 'I')
G.add_edge('G', 'H')


# In[ ]:


degrees = G.degree()


# In[ ]:


degree_values = sorted([d for n, d in degrees], reverse = True)


# In[ ]:


degree_values


# In[ ]:


histogram = [list(degree_values).count(i)/float(nx.number_of_nodes(G)) for i in degree_values]


# In[ ]:


plt.bar(degree_values, histogram)
plt.xlabel('Degree')
plt.ylabel('Fraction Of Nodes')
plt.show()


# In[4]:


P = nx.barabasi_albert_graph(10000,1)
degrees = P.degree()
degree_values = sorted([d for n, d in degrees], reverse = True)
histogram = [list(degree_values).count(i)/float(nx.number_of_nodes(P)) for i in degree_values]
plt.plot(degree_values, histogram, 'o')
plt.xlabel('Degree')
plt.ylabel('Fraction_Of_Nodes')
plt.xscale('log')
plt.yscale('log')
plt.show()


# In[5]:


P = nx.barabasi_albert_graph(1000,4)

print(nx.average_clustering(P))


# In[6]:


print(nx.average_shortest_path_length(P))


# In[11]:


P = nx.watts_strogatz_graph(1000,6, 0.04)
degrees = P.degree()
degree_values = sorted([d for n, d in degrees], reverse = True)
histogram = [list(degree_values).count(i)/float(nx.number_of_nodes(P)) for i in degree_values]
plt.bar(degree_values, histogram)
plt.xlabel('Degree')
plt.ylabel('Fraction_Of_Nodes')
plt.show()


# In[1]:


import networkx as nx
import operator


# In[2]:


G = nx.Graph()


# In[7]:


G.add_edge('A', 'B')
G.add_edge('A', 'D')
G.add_edge('A', 'E')
G.add_edge('B', 'D')
G.add_edge('B', 'C')
G.add_edge('C', 'D')
G.add_edge('C', 'F')
G.add_edge('E', 'F')
G.add_edge('E', 'G')
G.add_edge('F', 'G')
G.add_edge('G', 'H')
G.add_edge('G', 'I')


# In[8]:


G.nodes()


# In[9]:


comm_nei = [(e[0], e[1], len(list(nx.common_neighbors(G, e[0], e[1])))) for e in nx.non_edges(G)]


# In[11]:


sorted(comm_nei, key = operator.itemgetter(2), reverse = True)
print(comm_nei)

L = list(nx.jaccard_coefficient(G))
# In[14]:


L.sort(key = operator.itemgetter(2), reverse = True)
print(L)


# In[15]:


L = list(nx.resource_allocation_index(G))
L.sort(key = operator.itemgetter(2), reverse = True)
print(L)


# In[16]:


L = list(nx.adamic_adar_index(G))
L.sort(key = operator.itemgetter(2), reverse = True)
print(L)


# In[17]:


L = list(nx.preferential_attachment(G))
L.sort(key = operator.itemgetter(2), reverse = True)
print(L)


# In[19]:


G.node['A']['community'] = 0
G.node['B']['community'] = 0
G.node['C']['community'] = 0
G.node['D']['community'] = 0
G.node['E']['community'] = 1
G.node['F']['community'] = 1
G.node['G']['community'] = 1
G.node['H']['community'] = 1
G.node['I']['community'] = 1


# In[21]:


L = list(nx.cn_soundarajan_hopcroft(G))
L.sort(key = operator.itemgetter(2), reverse = True)
print(L)


# In[23]:


L = list(nx.ra_index_soundarajan_hopcroft(G))
L.sort(key = operator.itemgetter(2), reverse = True)
print(L)


# In[3]:


G.add_edge('A', 'D')
G.add_edge('A', 'C')
G.add_edge('A', 'E')
G.add_edge('C', 'G')
G.add_edge('G', 'D')
G.add_edge('D', 'B')
G.add_edge('D', 'E')
G.add_edge('D', 'H')
G.add_edge('H', 'F')
G.add_edge('E', 'H')


# In[5]:


comm_neig = [(e[0], e[1], len(list(nx.common_neighbors(G, e[0], e[1])))) for e in nx.non_edges(G)]


# In[8]:


sorted(comm_neig, key = operator.itemgetter(2), reverse = True)
print(comm_neig)


# In[ ]:


sorted(comm_nei, key = operator.itemgetter(2), reverse = True)
print(comm_nei)


# In[9]:


L = list(nx.jaccard_coefficient(G))
L.sort(key = operator.itemgetter(2), reverse = True)
L


# In[10]:


L = list(nx.resource_allocation_index(G))
L.sort(key = operator.itemgetter(2), reverse = True)
L


# In[11]:


L = list(nx.preferential_attachment(G))
L.sort(key = operator.itemgetter(2), reverse = True)
L


# In[12]:


G.node['A']['community'] = 0
G.node['B']['community'] = 0
G.node['C']['community'] = 0
G.node['D']['community'] = 0
G.node['E']['community'] = 1
G.node['F']['community'] = 1
G.node['G']['community'] = 0
G.node['H']['community'] = 1


# In[13]:


L = list(nx.cn_soundarajan_hopcroft(G))
L.sort(key = operator.itemgetter(2), reverse = True)
L


# In[14]:


L = list(nx.ra_index_soundarajan_hopcroft(G))
L.sort(key = operator.itemgetter(2), reverse = True)
L

