import psycopg2
import numpy as np

hostname = 'localhost'
username = 'postgres'
password = 'inter2010'
database = 'cloud_load'

conn = psycopg2.connect(host=hostname, user=username, password=password, dbname=database)

def read_data():
    cur1=conn.cursor()
    cur1.execute('select ts,num_of_req from calgary_http_60min order by ts')
    data=np.array(cur1.fetchall())
    return data