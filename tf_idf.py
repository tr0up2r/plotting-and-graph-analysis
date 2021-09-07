import pymysql.cursors
import pandas as pd
import numpy as np
import connect_to_db as cn
import json
from itertools import combinations
from sklearn.feature_extraction.text import TfidfVectorizer
from numba import jit, cuda


def extract_author(query_result):
    # dictionary 형태의 query 결과로부터 author만 추출해냄.
    tmp = json.dumps(query_result)
    tmp = tmp.split(':')
    tmp = tmp[1].split('}')
    res_author = tmp[0]

    return res_author


def extract_bodies(author, table):
    # 추출해 낸 author를 이용해 comment body만 끌어옴.
    connection = cn.make_connection('connection.csv')
    cursor = connection.cursor(pymysql.cursors.DictCursor)
    sql = f'select body from {table} where author = {author};'

    cursor.execute(sql)
    result = cursor.fetchall()
    connection.close()

    # 추출된 body 목록을 list 형태로 바꾸어서 return.
    result_df = pd.DataFrame(result)
    result_list = list(np.array(result_df['body'].tolist()))

    return result_list


@cuda.jit
def tf_idf_similarity(document):
    # document로부터 모든 combination 구하기.
    result_comb = list(combinations(document, 2))

    similarity = 0
    length = len(result_comb)

    for res in result_comb:
        doclist = list(res)

        # 문장이 아닌, 단어 comment는 제거.
        # tf-idf 검사 시 ValueError 발생.
        if len(doclist[0].split(' ')) > 1 and len(doclist[1].split(' ')) > 1:
            tfidf_vectorizer = TfidfVectorizer(min_df=1)
            tfidf_matrix = tfidf_vectorizer.fit_transform(doclist)

            document_distances = (tfidf_matrix * tfidf_matrix.T)

            similarity += document_distances.toarray()[0][1]

        else:
            length -= 1

    if length != 0:
        return similarity / length

    else:
        return 0


def update_similarity_column(author, similarity, column_name, connection, cursor):
    sql = f'update mentor set {column_name}={similarity} where author={author};'

    cursor.execute(sql)
    connection.commit()


if __name__ == "__main__":
    connection = cn.make_connection('connection.csv')

    cursor = connection.cursor(pymysql.cursors.DictCursor)

    sql = 'select author from mentor where comment_cnt > 1 and comment_cnt < 10 and is_valid = 1 and comment_similarity is null;'
    cursor.execute(sql)

    # result가 dict 형태로 return.
    result = cursor.fetchall()
    connection.close()

    author_similarity_list = []

    count = 0

    connection = cn.make_connection('connection.csv')
    cursor = connection.cursor(pymysql.cursors.DictCursor)

    for res in result:
        author = extract_author(res)
        bodies = extract_bodies(author, 'comments')
        similarity = tf_idf_similarity(bodies)

        update_similarity_column(author, similarity, 'comment_similarity', connection, cursor)

        count += 1
        print(count, end=' ')

    connection.close()