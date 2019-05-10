import psycopg2
import numpy as np

hostname = 'localhost'
username = 'postgres'
password = 'inter2010'
database = 'load_cloud'

conn = psycopg2.connect(host=hostname, user=username, password=password, dbname=database)

def read_data():
    cur1=conn.cursor()
    cur1.execute('select ts,num_of_req from nasa_http_1min '
                 ' order by ts offset 2000')
    data=np.array(cur1.fetchall())
    print('data read from DB!')
    return data