import psycopg2
import numpy as np

hostname = 'localhost'
username = 'postgres'
password = 'inter2010'
database = 'load_cloud'

conn = psycopg2.connect(host=hostname, user=username, password=password, dbname=database)
cur1=conn.cursor()
cur1.execute('select * from google_ram_req_100 order by ts')
ram_data = np.array(cur1.fetchall())
ts=ram_data[:,0]
ram=ram_data[:,1]
cur1.execute('select * from google_cpu_req_100 order by ts')
cpu_data = np.array(cur1.fetchall())
cpu=cpu_data[:,1]

import matplotlib.pyplot as plt
fig = plt.figure(facecolor='white',figsize=(12, 8))
ax = fig.add_subplot(211)
ax.plot(ts, cpu,color='red', label='CPU',alpha=0.85)
plt.legend()
plt.grid()
plt.ylabel('CPU Consumption')
ax = fig.add_subplot(212)
ax.plot(ts, ram,color='blue', label='RAM',alpha=0.85)
plt.legend()
plt.grid()
plt.ylabel('RAM Consumption')
plt.xlabel('Time Index')
#
plt.savefig('REAL_100.png', format='png', dpi=500)
plt.show()






