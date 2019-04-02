import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import math
from PyEMD import EMD
import time
from datetime import datetime

''' Read Raw Data '''
data = np.array(pd.read_csv('/home/vacek/Cloud/ThesisNew/Thesis/Data/NASA/NASA_access_log_Aug95/nasa.csv',sep ='- -'))
domains=data[:,0]
timestamps=[]
ids=[]
for k in range(len(domains)):
    time_dates=str(data[k,1]).split('-')[0][2:-1]
    timestamps.append(datetime.strptime(time_dates.replace('/','-').replace('Aug','08'), '%d-%m-%Y:%H:%M:%S'))
    ids.append(k+1)
print('total length of data is ',len(timestamps),len(ids))
del data

print('**************')
''' Set PWS = 5 min '''
pws=5
unique_ids_total=[]
unique_ids=[]
unique_ts=[]
unique_ts_total=[]
source=timestamps[0]
for k in range(len(timestamps)):
    td= timestamps[k]-source
    dif_min=int(round(td.total_seconds() / 60))
    #print(dif_min)
    if (dif_min)<pws:
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
        sample_time.append(unique_ts_total[i-2][0])
    sample_ts.append(i)
print(len(num_reqs),len(unique_ids_total),len(sample_time),len(sample_ts))
print('first time is ',sample_time[0])
print('last time is ',sample_time[-1])
plt.hist(num_reqs,bins='auto')
plt.xlabel('Number of Requests')
plt.ylabel('frequency')
plt.show()
plt.plot(sample_time,num_reqs,color='red',label='original data')
plt.xlabel('time')
plt.legend()
plt.grid()
plt.ylabel('Number of Requests')
plt.show()

''' writing raw data to db '''
import psycopg2

hostname = 'localhost'
username = 'postgres'
password = 'inter2010'
database = 'load_cloud'

conn = psycopg2.connect(host=hostname, user=username, password=password, dbname=database)
cur0 = conn.cursor()
cur0.execute('delete from nasa_http_5min')
cur0.execute('delete from nasa_http_emd_5min')
conn.commit()

print('writing raw data to DB ...')
cur1=conn.cursor()
for kkk in range(len(sample_time)):
        cur1.execute('insert into nasa_http_5min values (%s,%s,%s)',(int(kkk+1),int(num_reqs[kkk]),sample_time[kkk]))
        conn.commit()

print('****** Start to EDM Analysis ')
emd = EMD()
IMFs = emd(np.array(num_reqs,dtype=float))
print('****** 20min Analysis Completed! ')
num_of_IMF=len(IMFs)
print('total number of IMFs are',num_of_IMF)
imf_lens=[]

''' plotting IMFs'''
for imf_index in range(len(IMFs)):
    imf_lens.append(len(IMFs[imf_index]))
    for kkk in range(len(sample_time)):
            cur2 = conn.cursor()
            cur2.execute('insert into nasa_http_emd_5min values (%s,%s,%s,%s)',
                         (int(kkk + 1), int(IMFs[imf_index][kkk]), sample_time[kkk],int(imf_index+1)))
            conn.commit()
    print('IMF ' + str(imf_index + 1) + ' written to DB !!! ')
    plt.plot(sample_time, IMFs[imf_index])
    plt.title('imf # ' + str(imf_index + 1))
    plt.show()


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
















