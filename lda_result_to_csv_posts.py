# for text preprocessing
import re
import spacy

from nltk.corpus import stopwords 
from nltk.stem.wordnet import WordNetLemmatizer
import string

# import vectorizers
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer

# import numpy for matrix operation
import numpy as np

# import LDA from sklearn
from sklearn.decomposition import LatentDirichletAllocation

import connect_to_db as cn
import pandas as pd

from gensim import corpora
import gensim


# to suppress warnings
from warnings import filterwarnings
filterwarnings('ignore')

nlp = spacy.load('en_core_web_sm')

community = 7

sql = f'select node_id from nodes where community_id_fastgreedy_is = {community}'
result_df = cn.select_query_result_to_df(sql)
authors = np.array(result_df['node_id'].astype(str).values.tolist())

length = len(authors)

doc = []

for i in range(length):
    sql2 = f"select distinct p.post_key, p.title from posts p, comments c where p.post_key = c.link_key and c.author = '{authors[i]}' and c.link_key = c.parent_key and p.is_valid_author=1;";
    result_df2 = cn.select_query_result_to_df(sql2)
    if not result_df2.empty:
        titles = np.array(result_df2['title'].astype(str).values.tolist())
        doc.extend(titles)
        
corpus = doc

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

# clean data stored in a new list
clean_corpus = [clean(doc).split() for doc in corpus]

dictionary = corpora.Dictionary(clean_corpus)
corpus = [dictionary.doc2bow(text) for text in clean_corpus]

# build lda model.
ldamodel = gensim.models.LdaMulticore(corpus, id2word=dictionary, num_topics=20, passes=5)


def format_topics_sentences(ldamodel, corpus, texts):
    # Init output
    sent_topics_df = pd.DataFrame()

    # Get main topic in each document
    for i, row in enumerate(ldamodel[corpus]):
        row = sorted(row, key=lambda x: (x[1]), reverse=True)
        # Get the Dominant topic, Perc Contribution and Keywords for each document
        for j, (topic_num, prop_topic) in enumerate(row):
            if j == 0:  # => dominant topic
                wp = ldamodel.show_topic(topic_num)
                topic_keywords = ", ".join([word for word, prop in wp])
                sent_topics_df = sent_topics_df.append(pd.Series([int(topic_num), round(prop_topic,4), topic_keywords]), ignore_index=True)
            else:
                break
    sent_topics_df.columns = ['Dominant_Topic', 'Perc_Contribution', 'Topic_Keywords']

    # Add original text to the end of the output
    contents = pd.Series(texts)
    sent_topics_df = pd.concat([sent_topics_df, contents], axis=1)
    return(sent_topics_df)


df_topic_sents_keywords = format_topics_sentences(ldamodel=ldamodel, corpus=corpus, texts=doc)

# Format
df_dominant_topic = df_topic_sents_keywords.reset_index()
df_dominant_topic.columns = ['Document_No', 'Dominant_Topic', 'Topic_Perc_Contrib', 'Keywords', 'Text']

# write csv file
df_dominant_topic.to_csv(f'lda_result_gensim_community_{community}_posts.csv', sep=',', na_rep='NaN', index=None)