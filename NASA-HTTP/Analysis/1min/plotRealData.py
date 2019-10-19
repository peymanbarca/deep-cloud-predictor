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

cur0.execute('select ts,num_of_req from nasa_http_1min '
             ' '
             ' order by ts')
data = np.array(cur0.fetchall())

ts = data[:, 0]
num_req = data[:, 1]


fig = plt.figure(facecolor='white', figsize=(14, 8))
plt.plot(ts ,num_req,'-',  color='red',alpha=0.9)
plt.grid()
plt.xlabel('Time Index')
plt.ylabel('Number of Requests')
plt.savefig(''
            'Main-Signal' + '.png', dpi=700)
plt.pause(5)
plt.close()