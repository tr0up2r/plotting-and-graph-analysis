import igraph
import pandas as pd
import csv

df = pd.read_csv('mentor_mentee_relationship_until_november.csv')

data = df[['mentor', 'mentee', 'sum_es_score']]

# 그래프 생성
tuples = [tuple(x) for x in data.values]
g = igraph.Graph.TupleList(tuples, directed=True, edge_attrs=['sum_es_score'])

g.to_undirected(mode='collapse', combine_edges='sum')

dendrogram = g.community_fastgreedy()
clusters = dendrogram.as_clustering()
membership = clusters.membership

writer = csv.writer(open("community_fastgreedy_es_until_november.csv", "w", newline=''))
writer.writerow(['node_id', 'community_id'])
for name, membership in zip(g.vs["name"], membership):
    writer.writerow([name, membership])

