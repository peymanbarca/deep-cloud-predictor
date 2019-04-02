import psycopg2
import numpy as np

hostname = 'localhost'
username = 'postgres'
password = 'inter2010'
database = 'load_cloud'

conn = psycopg2.connect(host=hostname, user=username, password=password, dbname=database)

def read_data():
    cur1=conn.cursor()
    cur1.execute('select * from google_ram_req_200 order by ts')
    ram_data=np.array(cur1.fetchall())
    cur1.execute('select * from google_cpu_req_200 order by ts')
    cpu_data = np.array(cur1.fetchall())
    return ram_data,cpu_data