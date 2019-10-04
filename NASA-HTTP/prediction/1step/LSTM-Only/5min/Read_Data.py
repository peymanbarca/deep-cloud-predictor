import psycopg2
import numpy as np

hostname = 'localhost'
username = 'postgres'
password = 'inter2010'
database = 'load_cloud'

conn = psycopg2.connect(host=hostname, user=username, password=password, dbname=database)

def read_data():
    cur1=conn.cursor()
    cur1.execute('select ts,num_of_req from nasa_http_5min where num_req_pred is null order by ts offset 400')
    data=np.array(cur1.fetchall())
    print('data read from DB!')
    return data