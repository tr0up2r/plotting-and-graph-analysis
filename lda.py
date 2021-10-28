import spacy
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
import string
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
import numpy as np
from sklearn.decomposition import LatentDirichletAllocation
import csv
import connect_to_db as cn

# import nltk
# nltk.download('wordnet')

# to suppress warnings
from warnings import filterwarnings
filterwarnings('ignore')
nlp = spacy.load('en_core_web_sm')

community = 3

sql = f'select comment_key, body from comments c, nodes n where c.author = n.node_id and c.is_valid=1 and n.community_id_fastgreedy_is = {community};'
result_df = cn.select_query_result_to_df(sql)
corpus = list(np.array(result_df['body'].astype(str).values.tolist()))

comment_keys = list(np.array(result_df['comment_key'].astype(str).values.tolist()))

# stop loss words
stop = set(stopwords.words('english'))

# punctuation
exclude = set(string.punctuation)

# lemmatization
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

# Converting text into numerical representation
tf_idf_vectorizer = TfidfVectorizer(tokenizer=lambda doc: doc, lowercase=False)

# Converting text into numerical representation
cv_vectorizer = CountVectorizer(tokenizer=lambda doc: doc, lowercase=False)

# Array from TF-IDF Vectorizer
tf_idf_arr = tf_idf_vectorizer.fit_transform(clean_corpus)

# Array from Count Vectorizer
cv_arr = cv_vectorizer.fit_transform(clean_corpus)

# Creating vocabulary array which will represent all the corpus
vocab_tf_idf = tf_idf_vectorizer.get_feature_names()

# Creating vocabulary array which will represent all the corpus
vocab_cv = cv_vectorizer.get_feature_names()

# Create object for the LDA class
# Inside this class LDA: define the components:
lda_model = LatentDirichletAllocation(n_components=20, max_iter=10, random_state=20)

# fit transform on model on our count_vectorizer : running this will return our topics
X_topics = lda_model.fit_transform(tf_idf_arr)

# .components_ gives us our topic distribution
topic_words = lda_model.components_

#  Define the number of Words that we want to print in every topic : n_top_words
n_top_words = 5

topic_words_list = []

for i, topic_dist in enumerate(topic_words):

    # np.argsort to sorting an array or a list or the matrix acc to their values
    sorted_topic_dist = np.argsort(topic_dist)

    # Next, to view the actual words present in those indexes we can make the use of the vocab created earlier
    topic_words = np.array(vocab_tf_idf)[sorted_topic_dist]

    # so using the sorted_topic_indexes we ar extracting the words from the vocabulary
    # obtaining topics + words
    # this topic_words variable contains the Topics  as well as the respective words present in those Topics
    topic_words = topic_words[:-n_top_words:-1]
    topic_words_list.append(topic_words)

writer = csv.writer(open(f"lda_topics_community_{community}.csv", "w", newline=''))
writer.writerow(['topic', 'topic_words'])
for tw in enumerate (topic_words_list):
    writer.writerow(tw)

# To view what topics are assigned to the douments:

doc_topic = lda_model.transform(tf_idf_arr)

topic_list = []

# iterating over ever value till the end value
for n in range(doc_topic.shape[0]):

    # argmax() gives maximum index value
    topic_doc = doc_topic[n].argmax()

    topic_list.append(topic_doc)

writer = csv.writer(open(f"lda_result_community_{community}.csv", "w", newline=''))
writer.writerow(['comment_key', 'topic'])
for ck, topic in zip(comment_keys, topic_list):
    writer.writerow([ck, topic])