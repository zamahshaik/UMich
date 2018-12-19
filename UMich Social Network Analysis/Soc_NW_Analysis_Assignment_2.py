
# coding: utf-8

# ### Assignment 2 - Network Connectivity

# In[1]:


import networkx as nx

# This line must be commented out when submitting to the autograder
# !head email_network.txt


# In[2]:


# Question 1

# Using networkx, load up the directed multigraph from email_network.txt. 
# Make sure the node names are strings.

# This function should return a directed multigraph networkx graph.
def answer_one():
    
    G = nx.read_edgelist('email_network.txt', delimiter = '\t', data = [('time', int)],
                     create_using = nx.MultiDiGraph())
    
    return G

# answer_one()


# In[3]:


# Question 2

# How many employees and emails are represented in the graph from Question 1?

# This function should return a tuple (#employees, #emails).
def answer_two():
    
    G = answer_one()
    employees = len(G.nodes())
    emails = len(G.edges())
    
    return (employees, emails)

# answer_two()


# In[4]:


# Question 3

#     Part 1. Assume that information in this company can only be exchanged through email.

#     When an employee sends an email to another employee, a communication channel has been created, 
#     allowing the sender to provide information to the receiver, but not vice versa.

#     Based on the emails sent in the data, is it possible for information to go from every employee to every other employee?

#     Part 2. Now assume that a communication channel established by an email allows information to be exchanged both ways.

#     Based on the emails sent in the data, is it possible for information to go from every employee to every other employee?

# This function should return a tuple of bools (part1, part2).

def answer_three():
        
    G = answer_one()
    
    return (nx.is_strongly_connected(G), nx.is_connected(G.to_undirected()))

# answer_three()


# In[5]:


# Question 4

# How many nodes are in the largest (in terms of nodes) weakly connected 
# component?

# This function should return an int.

def answer_four():
        
    G = answer_one()
    
    return len(max(nx.weakly_connected_components(G), key=len))

# answer_four()


# In[6]:


# Question 5

# How many nodes are in the largest (in terms of nodes) strongly connected 
# component?

# This function should return an int

def answer_five():
        
    G = answer_one()
    
    return len(max(nx.strongly_connected_components(G), key = len))

# answer_five()


# In[7]:


# Question 6

# Using the NetworkX function strongly_connected_component_subgraphs, 
# find the subgraph of nodes in a largest strongly connected component. 
# Call this graph G_sc.

# This function should return a networkx MultiDiGraph named G_sc.

def answer_six():
        
    G = answer_one()
    G_sc = max(nx.strongly_connected_component_subgraphs(G), key=len)
    
    return G_sc

# answer_six()


# In[8]:


# Question 7

# What is the average distance between nodes in G_sc?

# This function should return a float.

def answer_seven():
        
    G_sc = answer_six()
    
    return nx.average_shortest_path_length(G_sc)

# answer_seven()


# In[9]:


# Question 8

# What is the largest possible distance between two employees in G_sc?

# This function should return an int.

def answer_eight():
        
    G_sc = answer_six()
    
    return nx.diameter(G_sc)

# answer_eight()


# In[10]:


# Question 9

# What is the set of nodes in G_sc with eccentricity equal to the diameter?

# This function should return a set of the node(s).

def answer_nine():
       
    G_sc = answer_six()
    
    return set(nx.periphery(G_sc))

# answer_nine()


# In[11]:


# Question 10

# What is the set of node(s) in G_sc with eccentricity equal to the radius?

# This function should return a set of the node(s).

def answer_ten():
        
    G_sc = answer_six()
    
    return set(nx.center(G_sc))

# answer_ten()


# In[55]:


# Question 11

# Which node in G_sc is connected to the most other nodes by a 
# shortest path of length equal to the diameter of G_sc?

# How many nodes are connected to this node?

# This function should return a 
# tuple (name of node, number of satisfied connected nodes).

def answer_eleven():
        
    G_sc = answer_six()

    d = nx.diameter(G_sc)
    p = nx.periphery(G_sc)
    max_nodes = 0

    for node in p:
        count = 0
        l = nx.shortest_path_length(G_sc, node)
        for k, v in l.items():
            if v == d:
                count += 1
            if count > max_nodes:
                res_node = node
                max_nodes = count
    
    return (res_node, max_nodes)

# answer_eleven()


# In[57]:


# Question 12

# Suppose you want to prevent communication from flowing to the node that you found in the previous question 
# from any node in the center of G_sc, what is the smallest number of nodes you would need to remove from the 
# graph (you're not allowed to remove the node from the previous question or the center nodes)?

# This function should return an integer.

def answer_twelve():
    G_sc = answer_six()

    c = nx.center(G_sc)[0]
    node = answer_eleven()[0]

    return len(nx.minimum_node_cut(G_sc, c, node))

# answer_twelve()


# In[47]:


# Question 13

# Construct an undirected graph G_un using G_sc (you can ignore the attributes).

# This function should return a networkx Graph.

def answer_thirteen():
    
    G_sc = answer_six()

    G_un = nx.Graph(G_sc.to_undirected())
    
    return G_un

# answer_thirteen()


# In[52]:


# Question 14

# What is the transitivity and average clustering coefficient of graph G_un?

# This function should return a tuple (transitivity, avg clustering).

def answer_fourteen():
    
    G_un = answer_thirteen()

    trans = nx.transitivity(G_un)

    avg_cl_coef = nx.average_clustering(G_un)
    
    return trans, avg_cl_coef

# answer_fourteen()

