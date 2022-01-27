import pandas as pd
import pymysql
import csv

def make_connection(cn_csv):
    cn = pd.read_csv('connection.csv')
    connection = pymysql.connect(host=cn['host'][0], port=int(cn['port'][0]),
                                 user=cn['user'][0], password=cn['password'][0],
                                 db=cn['db'][0], charset=cn['charset'][0])
    return connection


# select query를 날려서 가져온 result를 DataFrame으로 반환하는 function.
def select_query_result_to_df(sql):
    connection = make_connection('connection.csv')
    cursor = connection.cursor(pymysql.cursors.DictCursor)
    cursor.execute(sql)
    result = cursor.fetchall()
    connection.close()
    
    result_df = pd.DataFrame(result)
    
    return result_df


def write_csv_for_db_update(filename, fields, result_list):
    with open(filename, 'w', newline='') as f:
        write = csv.writer(f)
        write.writerow(fields)
        write.writerows(result_list)