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

cur0.execute('select ts,num_of_req from calgary_http_10min '
             ' '
             ' order by ts')
data = np.array(cur0.fetchall())

ts = data[:, 0]
num_req = data[:, 1]


fig = plt.figure(facecolor='white', figsize=(12, 10))
plt.plot(ts ,num_req,'-.', label='Real Data', color='orange')
plt.legend()
plt.xlabel('time symbol')
plt.ylabel('number of requests')
plt.savefig('/home/vacek/Cloud/cloud-predictor/Calgary-HTTP/Prediction/GANS-EMD/10min-smooth/resutls/Analysis'
            '/Main-Signal' + '.png', dpi=400)
plt.pause(5)
plt.close()