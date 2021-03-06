import sys
sys.path.append('../custom_library')

import pandas as pd
import numpy as np
import csv
import re

import connect_to_db as cn


def get_top_keywords(filename):

    with open(f'../keywords_analysis/csv/keywords/{filename}', newline='') as f:
        reader = csv.reader(f)
        keywords_pair = list(reader)
        
    top_keywords = []
    for pair in keywords_pair:
        top_keywords.append(pair[0])
        
    return keywords_pair, top_keywords


def top_keywords_main():
    normal_pair, normal_top_keywords = get_top_keywords('keywords_normal.csv')
    hub_pair, hub_top_keywords = get_top_keywords('keywords_hub.csv')

    hub_top_50 = set(hub_top_keywords[:452]) - set(normal_top_keywords[:452])
    normal_top_50 = set(normal_top_keywords[:452]) - set(hub_top_keywords[:452])

    hub_top_50_pair = []
    normal_top_50_pair = []

    for pair in hub_pair:
        for word in hub_top_50:
            if pair[0] == word:
                hub_top_50_pair.append(pair)
            
    for pair in normal_pair:
        for word in normal_top_50:
            if pair[0] == word:
                normal_top_50_pair.append(pair)            
            
    hub_top_50_df = pd.DataFrame(hub_top_50_pair)
    normal_top_50_df = pd.DataFrame(normal_top_50_pair)

    hub_top_50_df.to_csv(f"../keywords_analysis/csv/keywords/top_50_keywords_hub.csv", header=None, index=None)
    normal_top_50_df.to_csv(f"../keywords_analysis/csv/keywords/top_50_keywords_normal.csv", header=None, index=None)


def get_hub_and_normal_comments(parent_keys):
    hub_count = 0
    normal_count = 0
    
    with open(f'../keywords_analysis/csv/keywords/top_50_keywords_hub.csv', newline='') as f:
        reader = csv.reader(f)
        hub_pairs = list(reader)
    
    with open(f'../keywords_analysis/csv/keywords/top_50_keywords_normal.csv', newline='') as f:
        reader = csv.reader(f)
        normal_pairs = list(reader)
        
    hub_top_50 = []
    normal_top_50 = []
    
    for h_pair, n_pair in zip(hub_pairs, normal_pairs):
        hub_top_50.append(h_pair[0])
        normal_top_50.append(n_pair[0])
        
    keys_len = len(parent_keys)
    
    hub_keys_list = []
    normal_keys_list = []

    for i in range(keys_len):
        hub_keys = []
        normal_keys = []
        sql = f"select body from comments where is_valid=1 and is_valid_author=1 and comment_key='{parent_keys[i]}'"
        result_df = cn.select_query_result_to_df(sql)
        if result_df.empty:
            continue
        docs = list(np.array(result_df['body'].astype(str).values.tolist()))
        
        is_hub = False
        is_normal = False
        
        hub_words = []
        normal_words = []
        
        doc_words = docs[0].split(' ')
        
        for h_w, n_w in zip(hub_top_50, normal_top_50):
            if h_w in doc_words:
                is_hub = True
                hub_words.append(h_w)
            elif n_w in doc_words:
                is_normal = True
                normal_words.append(n_w)
            if is_hub and is_normal:
                break
       
        # print(hub_words)
        print(normal_words)
            
        if is_hub and not is_normal:
            hub_keys.append(parent_keys[i])
            hub_keys.extend(hub_words)
            hub_keys_list.append(hub_keys)
        elif is_normal and not is_hub:
            normal_keys.append(parent_keys[i])
            normal_keys.extend(normal_words)
            normal_keys_list.append(normal_keys)
        
    return hub_keys_list, normal_keys_list


def get_comment_keys_main():
    sql = "select parent_key, comment_key, body from comments where parent_key in (select comment_key from comments where link_key=parent_key) and is_valid=1 and is_valid_author=1;"
    result_df = cn.select_query_result_to_df(sql)

    parent_keys = list(np.array(result_df['parent_key'].astype(str).values.tolist()))
    docs = list(np.array(result_df['body'].astype(str).values.tolist()))

    hub_keys, normal_keys = get_hub_and_normal_comments(parent_keys)

    hub_keys_df = pd.DataFrame(hub_keys)
    normal_keys_df = pd.DataFrame(normal_keys)

    hub_keys_df.to_csv(f"../keywords_analysis/csv/keys/hub_comment_keys_and_words.csv", header=None, index=None)
    normal_keys_df.to_csv(f"../keywords_analysis/csv/keys/normal_comment_keys_and_words.csv", header=None, index=None)
    
    
if __name__ == "__main__":
    get_comment_keys_main()