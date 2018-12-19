
# coding: utf-8

# In[9]:


import networkx as nx
import pandas as pd
import numpy as np
from networkx.algorithms import bipartite


# This is the set of employees
employees = set(['Pablo',
                 'Lee',
                 'Georgia',
                 'Vincent',
                 'Andy',
                 'Frida',
                 'Joan',
                 'Claude'])

# This is the set of movies
movies = set(['The Shawshank Redemption',
              'Forrest Gump',
              'The Matrix',
              'Anaconda',
              'The Social Network',
              'The Godfather',
              'Monty Python and the Holy Grail',
              'Snakes on a Plane',
              'Kung Fu Panda',
              'The Dark Knight',
              'Mean Girls'])


# you can use the following function to plot graphs
# make sure to comment it out before submitting to the autograder
def plot_graph(G, weight_name=None):
    '''
    G: a networkx G
    weight_name: name of the attribute for plotting edge weights (if G is weighted)
    '''
#     %matplotlib notebook
#     import matplotlib.pyplot as plt
    
    plt.figure()
    pos = nx.spring_layout(G)
    edges = G.edges()
    weights = None
    
    if weight_name:
        weights = [int(G[u][v][weight_name]) for u,v in edges]
        labels = nx.get_edge_attributes(G,weight_name)
        nx.draw_networkx_edge_labels(G,pos,edge_labels=labels)
        nx.draw_networkx(G, pos, edges=edges, width=weights);
    else:
        nx.draw_networkx(G, pos, edges=edges);


# In[2]:


# Question 1

# Using NetworkX, load in the bipartite graph from Employee_Movie_Choices.txt and return that graph.

# This function should return a networkx graph with 19 nodes and 24 edges

def answer_one():
        
    movie_choices = nx.read_edgelist('Employee_Movie_Choices.txt', delimiter = "\t")
    
    return movie_choices

# plot_graph(answer_one())


# In[3]:


# Question 2

# Using the graph from the previous question, add nodes attributes named 'type' 
# where movies have the value 'movie' and employees have the value 'employee' and return that graph.

# This function should return a networkx graph with node attributes {'type': 'movie'} or {'type': 'employee'}

def answer_two():
    
    P = answer_one()        
    for item,value in P.nodes(data=True):                
        if item in movies:                        
            value['type']='movie'                
        if item in employees:                        
            value['type']='employee' 
    return P


# In[4]:


# Question 3

# Find a weighted projection of the graph from answer_two which tells us 
# how many movies different pairs of employees have in common.

# This function should return a weighted projected graph.

def answer_three():
        
    P = answer_two()
    weig_proj = bipartite.weighted_projected_graph(P, employees)
    
    return weig_proj

# plot_graph(answer_three())


# In[5]:


# Question 4

# Suppose you''d like to find out if people that have a high relationship score also like the same types of movies.

# Find the Pearson correlation ( using DataFrame.corr() ) between employee relationship scores 
# and the number of movies they have in common. 
# If two employees have no movies in common it should be treated as a 0, not a missing value, 
# and should be included in the correlation calculation.

# This function should return a float.

def answer_four():
        
    B = answer_three()
    B_df = pd.DataFrame(list(B.edges(data = True)), columns = ['From1', 'To1', 'mov_score'])

    EmpRel = nx.read_edgelist('Employee_Relationships.txt', data = [('rel_score', int)])
    EmpRel_df = pd.DataFrame(list(EmpRel.edges(data = True)), columns=['From', 'To', 'rel_score'])

    # Convert the one edge direction to double directional
    B_copy_df = B_df.copy()
    B_copy_df.rename(columns = {'From1': 'To', 'To1': 'From'}, inplace = True)
    B_df.rename(columns = {'From1': 'From', 'To1': 'To'}, inplace = True)
    B_final_df = pd.concat([B_df, B_copy_df])

    #Merge Weighted Bi-directional and Emp Relation DFs
    final_df = pd.merge(B_final_df, EmpRel_df, on = ['From', 'To'], how = 'right')

    final_df['mov_score'] = final_df['mov_score'].map(lambda x: x['weight'] if type(x) == dict else None)
    final_df['mov_score'].fillna(value = 0, inplace = True)
    final_df['rel_score'] = final_df['rel_score'].map(lambda x: x['rel_score'])

    corr_val = final_df['mov_score'].corr(final_df['rel_score'])
    
    return corr_val

# answer_four()

