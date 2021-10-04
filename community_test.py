import igraph
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from igraph import *
import connect_to_db as cn
import csv


g = igraph.Graph.Read_Ncol('network.txt')

g.to_undirected(mode='collapse', combine_edges='sum')

'''
dendrogram = g.community_fastgreedy()
clusters = dendrogram.as_clustering()
membership = clusters.membership
'''


# dendrogram = g.community_edge_betweenness()
clusters = g.community_leiden()
# clusters = dendrogram.as_clustering()
membership = clusters.membership

writer = csv.writer(open("output3.csv", "w", newline=''))
writer.writerow(['node_id', 'community_id'])
for name, membership in zip(g.vs["name"], membership):
    writer.writerow([name, membership])