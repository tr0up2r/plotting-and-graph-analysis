# For using custom module.
import sys
sys.path.append('../custom_library')

import igraph
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from igraph import *
import connect_to_db as cn
import parmap


def component_main(index):
    if index % 100000 == 0:
        start_index = index - 100000
    else:
        start_index = index - (index % 100000)
    data = pd.read_csv('../graph/csv/mentor_mentee_relationship.csv')

    data1 = data[['mentor', 'mentee', 'average_is_score']]
    tuples1 = [tuple(x) for x in data1.values]
    g1 = igraph.Graph.TupleList(tuples1, directed=True, edge_attrs=['average_is_score'])

    s_components = g1.clusters()
    s_component_list = []
    snum_c = len(s_components)
    print(snum_c)

    for i in range(start_index, index):
        lc = len(s_components[i])
        print(i, end= ' ')
        if lc < 2:
            continue
        print(lc, end=' ')
        s_component_list.append([i, lc])
    
    fields = ['index', 'component_len']
    cn.write_csv_for_db_update(f"../graph/csv/strongly_connected_components_{index}.csv", fields, s_component_list)
    
    # component_df = pd.DataFrame(component_list)
    # component_df.to_csv(f"../graph/csv/{index}_strong_component.csv", header=['index', 'component_len'], index=None)
    
    
index_list = [100000, 200000, 300000, 400000, 500000, 600000, 621472]

if __name__ == '__main__':
    # multi processing.
    parmap.map(component_main, index_list, pm_pbar=True, pm_processes=7)