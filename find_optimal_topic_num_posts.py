import spacy

from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
import string

import numpy as np
import connect_to_db as cn

import gensim
import csv

from gensim import corpora
from gensim.models.coherencemodel import CoherenceModel


# to suppress warnings
from warnings import filterwarnings
filterwarnings('ignore')
nlp = spacy.load('en_core_web_sm')

community = 0

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

coherences_perplexities = []
num_topics=[5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90, 95, 100]

for i in num_topics:
    ntopics = i

    # ldamodel = gensim.models.ldamodel.LdaModel(corpus, id2word=dictionary, num_topics=ntopics, passes=5)
    ldamodel = gensim.models.LdaMulticore(corpus, id2word=dictionary, num_topics=ntopics, passes=5)

    cm = CoherenceModel(model=ldamodel, corpus=corpus, coherence='u_mass')
    coherences_perplexities.append([ntopics, cm.get_coherence(), ldamodel.log_perplexity(corpus)])

fields = ['topic_num', 'coherence', 'perplexity']
cn.write_csv_for_db_update(f'community_{community}_coherences_perplexities_posts.csv', fields, coherences_perplexities)