import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import psycopg2
from sklearn.metrics import mean_squared_error
from math import sqrt
import itertools

hostname = 'localhost'
username = 'postgres'
password = 'inter2010'
database = 'load_cloud'

conn = psycopg2.connect(host=hostname, user=username, password=password, dbname=database)
cur0=conn.cursor()



total_req=[]
total_req_smooth=[]

for imf_index in range(1,21):
    print(imf_index)
    cur0.execute('select ts,num_of_req from saskatchewan_http_emd_10min '
                 ' where imf_index={} '
                 ' order by ts'.format(imf_index))
    data = np.array(cur0.fetchall())

    ts = data[:, 0]
    num_req = data[:, 1]

    total_req.append(list(num_req))

main_total_req__=np.zeros(len(ts))
for k in total_req:
    main_total_req__+=np.array(k,dtype=float)

for imf_index in range(1,21):
    print(imf_index)
    cur0.execute('select ts,num_of_req from saskatchewan_http_emd_10min_copy '
                 ' where imf_index={} '
                 ' order by ts'.format(imf_index))
    data = np.array(cur0.fetchall())

    ts = data[:, 0]
    num_req = data[:, 1]

    total_req_smooth.append(list(num_req))

main_total_req_smooth__=np.zeros(len(ts))
for k in total_req_smooth:
    main_total_req_smooth__+=np.array(k,dtype=float)

fig = plt.figure(facecolor='white', figsize=(14, 12))
plt.subplot(211)
plt.plot(ts ,main_total_req__,'-.', label='Real Data', color='orange')
plt.legend()
plt.xlabel('time symbol')
plt.ylabel('number of requests')
plt.subplot(212)
plt.plot(ts ,main_total_req_smooth__,'-.', label='Real Smoothed Data', color='brown')
plt.legend()
plt.xlabel('time symbol')
plt.ylabel('number of requests')
plt.savefig('/home/vacek/Cloud/cloud-predictor/Saskatchewan/prediction/GANS-EMD/10min-smooth/resutls/Analysis'
            '/Main-Signal-EMDs' + '.png', dpi=700)
plt.pause(5)
plt.close()