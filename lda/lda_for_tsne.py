# For using custom module.
import sys
sys.path.append('../custom_library')

import spacy
from nltk.corpus import stopwords 
from nltk.stem.wordnet import WordNetLemmatizer
import string
import numpy as np
import pandas as pd 
import connect_to_db as cn
from gensim import corpora
import gensim
import csv
import parmap

# to suppress warnings
from warnings import filterwarnings
filterwarnings('ignore')

nlp = spacy.load('en_core_web_sm')

# stop loss words 
stop = set(stopwords.words('english'))

# punctuation, 구두점 제거.
exclude = set(string.punctuation) 

# lemmatization, 표제어 추출. (am, are, is -> be, ed, s 등 제거.)
lemma = WordNetLemmatizer() 


# One function for all the steps:
def clean(doc):
    
    # convert text into lower case + split into words
    stop_free = " ".join([i for i in doc.lower().split() if i not in stop])
    
    # remove any stop words present
    punc_free = ''.join(ch for ch in stop_free if ch not in exclude)  
    
    # remove punctuations + normalize the text
    normalized = " ".join(lemma.lemmatize(word) for word in punc_free.split())
                
    return normalized


custom_stop_words = ["im", "going", "would", "like", "cant", "donâ€™t", "canâ€™t", "iâ€™ve", "iâ€™m", "me", "someone", "whatâ€™s", "it", "really", "feel", "live", "like", "fucking", "myself", "another", "help", "got", "get", "dont", "want", "anymore", "know", "make", "self", "everything", "see", "else", "oh", "there", "thing", "wanna", "wouldnâ€™t", "might", "itâ€™s", "didnâ€™t", "yâ€™all", "do", "anyone", "people", "ever", "please"]


def remove_custom_stop_words(word_lists):
    for word_list in word_lists:
        stops = []
        for word in word_list:
            # 단어가 custom stop words에 속하거나, 숫자거나, 알파벳 하나일 경우 제거.
            if word in custom_stop_words or word.isdigit() or len(word) == 1:
                stops.append(word)
        
        for stop in stops:
            word_list.remove(stop)
            
        # list가 stop words 제거로 인해 비었는지 확인.
        if not word_list:
            word_lists.remove(word_list)

    return word_lists


def save_topic_words_and_weights(table_name, community, count, remove_sw, same_topic_num):
    sql = f'select node_id from {table_name} where community_id_fastgreedy_is = {community}'
    result_df = cn.select_query_result_to_df(sql)
    authors = np.array(result_df['node_id'].astype(str).values.tolist())

    length = len(authors)

    doc = []

    for i in range(length):
        # sql2 = f"select distinct p.post_key, p.title from posts p, comments c where p.post_key = c.link_key and c.author = '{authors[i]}' and c.link_key = c.parent_key and p.is_valid_author=1 and MONTH(p.created_utc) <> 12;";
        # sql2 = f"select body from comments where author = '{authors[i]}' and is_valid=1 and link_key = parent_key;"
        sql2 = f"select distinct p.post_key, p.title from posts p, comments c where p.post_key = c.link_key and c.author = '{authors[i]}' and c.link_key = c.parent_key and p.is_valid_author=1;"
        result_df2 = cn.select_query_result_to_df(sql2)
        if not result_df2.empty:
            titles = np.array(result_df2['title'].astype(str).values.tolist())
            # titles = np.array(result_df2['body'].astype(str).values.tolist())
            doc.extend(titles)
            
    if len(doc) < 2:
        return None
        
    corpus = doc
    num_words = 50
    folder = 'topic_words'
        
    # clean data stored in a new list
    clean_corpus = [clean(doc).split() for doc in corpus]
    # custom stop words 제거.
    if remove_sw:
        clean_corpus = remove_custom_stop_words(clean_corpus)
        num_words = 10
        # num_words = 40
        folder = 'topic_words_stop_words_removed'
    dictionary = corpora.Dictionary(clean_corpus)
    corpus = [dictionary.doc2bow(text) for text in clean_corpus]
        
    num_topics = 1
    
    if same_topic_num:
        folder += '_same_topic_num'
        num_words = 20
    
    if not same_topic_num:
        if count >= 10000:
            num_topics = 10
        elif count >= 1000:
            num_topics = 5
        elif count >= 100:
            num_topics = 4
        elif count >= 10:
            num_topics = 3
        else:
            num_topics = 1
     
    # 결과가 매번 다르게 나오는 것을 방지하기 위한 seed 고정.
    SOME_FIXED_SEED = 624
    np.random.seed(SOME_FIXED_SEED)
    
    # Exception of empty list
    if corpus and dictionary:
        print(community, end=' ')
        ldamodel = gensim.models.LdaMulticore(corpus, id2word=dictionary, num_topics=num_topics, passes=10)
        x=ldamodel.show_topics(num_topics=num_topics, num_words=num_words,formatted=False)
        topics_words = [[wd[0] for wd in tp[1]] for tp in x]
        topics_words_weights = [[wd[1] for wd in tp[1]] for tp in x]    
    
        words_df = pd.DataFrame(topics_words)
        weights_df = pd.DataFrame(topics_words_weights)
        words_df.to_csv(f"../lda/csv/lda_results/{table_name}/posts/{folder}_{num_words}_for_tsne/community_{community}_topics_{num_words}_words.csv", header=None, index=None)
        # weights_df.to_csv(f"../lda/csv/lda_results/{table_name}/posts/{folder}_weights_{num_words}_for_tsne/community_{community}_topics_{num_words}_weights.csv", header=None, index=None)
        return community
    
    
sql = "select community_id_fastgreedy_is, count(*) from nodes group by community_id_fastgreedy_is having count(*) > 2 order by count(*) asc;"
result_df = cn.select_query_result_to_df(sql)
communities = list(np.array(result_df['community_id_fastgreedy_is'].values.tolist()))
counts = list(np.array(result_df['count(*)'].values.tolist()))

valid_communities = []
for community, count in zip(communities, counts):
    valid_community = save_topic_words_and_weights('nodes', community, count, True, True)
    if valid_community != None:
        valid_communities.append(valid_community)