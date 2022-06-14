# For using custom module.
import sys
sys.path.append('../custom_library')

import numpy as np
import pandas as pd 
import connect_to_db as cn
import yake
import csv


def extract_keywords(doc):
    kw_extractor = yake.KeywordExtractor()
    language = "en"
    max_ngram_size = 1
    deduplication_threshold = 0.9
    numOfKeywords = 5
    custom_kw_extractor = yake.KeywordExtractor(lan=language, n=max_ngram_size, dedupLim=deduplication_threshold, top=numOfKeywords, features=None)
    
    keyword_list = []
    for text in doc:
        keywords = custom_kw_extractor.extract_keywords(text)
        for kw in keywords:
            if kw[0].isalnum() and 'â' not in kw[0]:
                keyword_list.append(kw[0].lower())
        
    return keyword_list


def get_doc(table_name, author, degree):
    doc = []

    if degree == 'outdegree':
        sql = f"select distinct body from comments where author='{author}' and link_key = parent_key and is_valid_author=1 and is_valid=1;"
    else:
        sql = 'test'
    result_df = cn.select_query_result_to_df(sql)
    if not result_df.empty:
        if degree == 'outdegree':
            bodies = np.array(result_df['body'].astype(str).values.tolist())
            doc.extend(bodies)
    
    return doc


def yake_main(target):
    keywords_list = []
    total_comment_count = 0
    
    if target == 'hub':
        sql = "select distinct(body) from comments where link_key=parent_key and is_valid_author=1 and is_valid=1 and author in (select node_id from nodes where top_k_outdegree=0.1);"    
    else:
        sql = "select distinct(body) from comments where link_key=parent_key and is_valid_author=1 and is_valid=1 and author not in (select node_id from nodes where top_k_outdegree=0.1) limit 212826;"
    result_df = cn.select_query_result_to_df(sql)
        
    docs = list(np.array(result_df['body'].astype(str).values.tolist()))
    keywords_list = extract_keywords(docs)
   
    key_set = list(set(list(keywords_list)))
    counts = [0] * len(key_set)

    all_keywords = keywords_list
    print(f'[{target}] number of comments: {len(docs)}, number of keywords: {len(all_keywords)}')

    for i in range(len(all_keywords)):
        for j in range(len(key_set)):
            if all_keywords[i] == key_set[j]:
                counts[j] += 1
                break
                
    word_count_pair = []
    for word, count in zip(key_set, counts):
        word_count_pair.append([word, count])

    pair_dict = dict(word_count_pair)
    sorted_pair_dict = sorted(pair_dict.items(), key = lambda item: item[1], reverse = True)

    sorted_pair = []
    for pair in sorted_pair_dict:
        sorted_pair.append([pair[0], pair[1]])

    keywords_df = pd.DataFrame(sorted_pair)
    keywords_df.to_csv(f"../keywords_analysis/csv/keywords/keywords_{target}.csv", header=None, index=None)
    
    
if __name__ == "__main__":
    yake_main('hub')
    yake_main('normal')