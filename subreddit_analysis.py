import pymysql.cursors
import pandas as pd
import numpy as np
import connect_to_db as cn
from scipy.stats import entropy
import math
import csv
import parmap
import random

sql = "select distinct community_id_fastgreedy_is community_id from nodes order by community_id;"
communities_df = cn.select_query_result_to_df(sql)
communities = list(np.array(communities_df['community_id'].tolist()))

# multi processing 시, 병목 현상을 최소화하기 위해 community_id를 shuffle.
random.sample(communities, len(communities))
communities = sorted(communities, key=lambda k: random.random())


def combination(n, r):
    f = math.factorial
    return f(n) // f(r) // f(n-r)


def subreddit_per_community_main(index):
    if index % 2000 == 0:
        start_index = index - 2000
    else:
        start_index = index - (index % 2000)
    
    result_for_csv = []
    
    for i in range(start_index, index):
        sql = f"select subreddit_key, cast(sum(cnt) as signed) sum from (select subreddit_key, count(subreddit_key) as cnt from comments c inner join nodes n on c.author = n.node_id where n.community_id_fastgreedy_is = {communities[i]} and c.link_key = c.parent_key and c.is_valid = 1 group by subreddit_key union all select subreddit_key, count(subreddit_key) cnt from posts p inner join nodes n on p.author = n.node_id where n.community_id_fastgreedy_is = {communities[i]} and p.is_valid=1 group by subreddit_key) a group by subreddit_key;"
        subreddit_df = cn.select_query_result_to_df(sql)
        subreddit = list(np.array(subreddit_df['subreddit_key'].tolist()))
        subreddit_count = list(np.array(subreddit_df['sum'].tolist()))

        # The number of subreddit by community
        subreddits = len(subreddit)

        # The number of comments and posts by subreddit by community
        # 하나의 subreddit에만 comments, posts가 작성되었으면 entropy는 0으로 set.
        if subreddits == 1:
            subreddit_entropy = 0
        else:
            subreddit_entropy = entropy(subreddit_count, base=subreddits)

        # idk value name...
        combination_sum = 0
        for count in subreddit_count:
            if count > 1:
                combination_sum += combination(count, 2)
        value = combination_sum / combination(sum(subreddit_count), 2)
        
        result_for_csv.append([communities[i], subreddits, subreddit_entropy, value])
        
    fields = ['community_id', 'subreddits', 'subreddit_entropy', 'value']
    cn.write_csv_for_db_update(f"/home/mykim/source/plotting-and-graph-analysis/subreddit/subreddit_analysis_{index}.csv", fields, result_for_csv)
    
    
index_list = [2000, 4000, 6000, 8000, 10000, 12000, 14000, 16000, 18000, 20000, 21370]

if __name__ == '__main__':
    # multi processing
    parmap.map(subreddit_per_community_main, index_list, pm_pbar=True, pm_processes=11)