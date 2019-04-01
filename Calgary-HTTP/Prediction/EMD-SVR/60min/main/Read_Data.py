import psycopg2
import numpy as np

hostname = 'localhost'
username = 'postgres'
password = 'inter2010'
database = 'load_cloud'

conn = psycopg2.connect(host=hostname, user=username, password=password, dbname=database)

def read_data(imf_num):
    cur1=conn.cursor()
    cur1.execute('select ts,num_of_req from calgary_http_emd_60min where imf_index=%s and num_req_pred is null'
                 ' order by ts',([int(imf_num)]))
    data=np.array(cur1.fetchall())
    print('data read from DB!')
    return data