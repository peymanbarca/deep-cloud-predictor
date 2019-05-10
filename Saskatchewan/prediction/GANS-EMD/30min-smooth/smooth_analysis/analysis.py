import psycopg2
import numpy as np
from matplotlib import pyplot as plt

hostname = 'localhost'
username = 'postgres'
password = 'inter2010'
database = 'load_cloud'

conn = psycopg2.connect(host=hostname, user=username, password=password, dbname=database)
cur1=conn.cursor()

imf_num=3


cur1.execute('select ts,num_of_req from saskatchewan_http_emd_30min where imf_index=%s and num_req_pred is null'
             ' order by ts limit 300',([int(imf_num)]))
data=np.array(cur1.fetchall())
ts=data[:,0]
req=data[:,1]

cur1.execute('select ts,num_of_req from saskatchewan_http_emd_30min_copy where imf_index=%s and num_req_pred is null'
             ' order by ts limit 300',([int(imf_num)]))
data_smooth=np.array(cur1.fetchall())
req_smooth=data_smooth[:,1]



plt.plot(ts, req, label='real data')
plt.plot(ts, req_smooth, 'green', label='smooth data ')
plt.legend()
plt.show()

