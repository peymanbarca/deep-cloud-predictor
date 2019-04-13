import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import math
from PyEMD import EMD
import time
from datetime import datetime

''' Read Raw Data '''

''' EPA HTTP logs is 2 days 29 & 30 Aguest '''


data = np.array(pd.read_csv('/home/vacek/Cloud/ThesisNew/Thesis/Data/epa-http.txt/epa-http.csv',sep ='['))
domains=data[:,1]
timestamps=[]
ids=[]
for k in range(len(domains)):
    try:
        time_dates=str(domains[k])[:11]
        time_dates=time_dates.replace(time_dates[:3],time_dates[:2]+'-08-1995:')
        #timestamps.append(time_dates)
        timestamps.append(datetime.strptime(time_dates, '%d-%m-%Y:%H:%M:%S'))
    except:
        continue
timestamps=sorted(timestamps)
ids=list(np.linspace(1,len(timestamps),len(timestamps)))
print('total length of data is ',len(timestamps),len(ids))
del data

print('**************')
''' Set PWS = 5 sec '''
pws=5
unique_ids_total=[]
unique_ids=[]
unique_ts=[]
unique_ts_total=[]
source=timestamps[0]
for k in range(len(timestamps)):
    td= timestamps[k]-source
    dif_sec=int(round(td.total_seconds()))
    if (dif_sec)<pws:
        unique_ids.append(ids[k])
        unique_ts.append(source)
    else:
        unique_ids_total.append(unique_ids)
        unique_ts_total.append(unique_ts)
        unique_ids = []
        unique_ts=[]
        source=timestamps[k]
print(len(unique_ids_total),len(unique_ts_total))
print('*********')

''' extract clusters of  data '''
num_reqs=[]
sample_time=[]
sample_ts=[]
i=0
for k in unique_ids_total:
    i+=1
    num_reqs.append(len(k))
    try:
        sample_time.append(unique_ts_total[i-1][0])
    except:
        try:
            sample_time.append(unique_ts_total[i-2][0])
        except:
            sample_time.append(None)
    sample_ts.append(i)
print(len(num_reqs),len(unique_ids_total),len(sample_time),len(sample_ts))

print('first time is ',sample_time[0])
print('last time is ',sample_time[-1])
fig = plt.figure(facecolor='white',figsize=(10, 8))
plt.hist(num_reqs,bins='auto')
plt.xlabel('Number of Requests')
plt.ylabel('frequency')
#plt.savefig('figs/10sec/' + 'histogram.png', dpi=600)
plt.pause(3)
plt.close()
fig = plt.figure(facecolor='white',figsize=(10, 8))
plt.plot(num_reqs,color='red',label='original data')
plt.xlabel('time')
plt.legend()
plt.grid()
plt.ylabel('Number of Requests')
#plt.savefig('figs/10sec/' + 'original.png', dpi=600)
plt.pause(3)
plt.close()
#
''' writing raw data to db '''
import psycopg2

hostname = 'localhost'
username = 'postgres'
password = 'inter2010'
database = 'load_cloud'

conn = psycopg2.connect(host=hostname, user=username, password=password, dbname=database)
cur0 = conn.cursor()
cur0.execute('delete from epa_http_5sec')
cur0.execute('delete from epa_http_emd_5sec')
conn.commit()

print('writing raw data to DB ...')
cur1=conn.cursor()
for kkk in range(len(sample_time)):
        cur1.execute('insert into epa_http_5sec values (%s,%s,%s)',(int(kkk+1),int(num_reqs[kkk]),sample_time[kkk]))
        conn.commit()
#
print('****** Start to EDM Analysis ')
emd = EMD()
IMFs = emd(np.array(num_reqs,dtype=float))
print('****** 60min Analysis Completed! ')
num_of_IMF=len(IMFs)
print('total number of IMFs are',num_of_IMF)
imf_lens=[]

''' plotting IMFs'''
for imf_index in range(len(IMFs)):
    imf_lens.append(len(IMFs[imf_index]))
    for kkk in range(len(num_reqs)):
            cur2 = conn.cursor()
            cur2.execute('insert into epa_http_emd_5sec values (%s,%s,%s,%s)',
                         (int(kkk + 1), int(IMFs[imf_index][kkk]), sample_time[kkk],int(imf_index+1)))
            conn.commit()
    print('IMF ' + str(imf_index + 1) + ' written to DB !!! ')
    fig = plt.figure(facecolor='white', figsize=(10, 8))
    plt.plot(IMFs[imf_index])
    plt.title('imf # ' + str(imf_index + 1))
    plt.grid()
    #plt.savefig('figs/10sec/'+ 'IMF_' + str(imf_index + 1) + '.png', dpi=600)
    plt.pause(3)
    plt.close()


print('length of IMFs are',imf_lens)
print('***********')
''' reconstructing main data from its IMFs'''
reconstructed_data=np.zeros([1,imf_lens[0]])
for imf_index in range(len(IMFs)):
    reconstructed_data=reconstructed_data+np.array(IMFs[imf_index])
print('RMSE between original and reconstructed signal is',np.sqrt(np.sum(np.square(reconstructed_data-num_reqs))))
plt.subplot(2,1,1)
plt.plot(sample_time,num_reqs,color='blue',label='original data')
plt.subplot(2,1,2)
plt.plot(sample_time,num_reqs,color='red',label='reconstructed data')
plt.legend()
plt.grid()
plt.show()

plt.plot(sample_time,list(np.transpose(np.square(reconstructed_data-num_reqs))),color='blue',label='squared error')
plt.legend()
plt.grid()
plt.show()
















