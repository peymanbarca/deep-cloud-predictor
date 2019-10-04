import psycopg2
import numpy as np
from matplotlib import pyplot as plt

hostname = 'localhost'
username = 'postgres'
password = 'inter2010'
database = 'load_cloud'

conn = psycopg2.connect(host=hostname, user=username, password=password, dbname=database)

def read_data(imf_num):
    cur1=conn.cursor()
    cur1.execute('select ts,num_of_req from nasa_http_emd_10min where imf_index=%s and num_req_pred is null'
                 ' order by ts offset 150',([int(imf_num)]))
    data=np.array(cur1.fetchall())
    return data

reqs=[]
for i in range(1,16):
    data=read_data(i)
    ts=data[:,0]
    req=data[:,1]
    reqs.append(req)

real_req=np.zeros(len(req))
for k in reqs:
    real_req+=np.array(k)

print(len(real_req),len(ts))

fig = plt.figure(facecolor='white',figsize=(11, 6.5))
fig.tight_layout()
ax = fig.add_subplot(6,1,1)
plt.plot(ts,real_req,color='red',label='Real Requests')
plt.legend()
plt.ylabel('Number of Requests',size=7)
ax = fig.add_subplot(6,1,2)
plt.plot(ts,reqs[0],color='blue',label='IMF 1')
plt.legend()
ax = fig.add_subplot(6,1,3)
plt.plot(ts,reqs[1],color='blue',label='IMF 2')
plt.legend()
ax = fig.add_subplot(6,1,4)
plt.plot(ts,reqs[2],color='blue',label='IMF 3')
plt.legend()
ax = fig.add_subplot(6,1,5)
plt.plot(ts,reqs[13],color='blue',label='IMF 14')
plt.legend()
ax = fig.add_subplot(6,1,6)
plt.plot(ts,reqs[14],color='blue',label='Residual')
plt.legend()
plt.xlabel('Time Index')
fig.subplots_adjust(hspace=0.5)
plt.savefig('results/imf-shapes/'+str('imf-shapes') + '.png', dpi=600)
plt.show()
