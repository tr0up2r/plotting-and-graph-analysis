import pandas as pd
import pymysql

def make_connection(cn_csv):
    cn = pd.read_csv('connection.csv')
    connection = pymysql.connect(host=cn['host'][0], port=int(cn['port'][0]),
                                 user=cn['user'][0], password=cn['password'][0],
                                 db=cn['db'][0], charset=cn['charset'][0])
    return connection