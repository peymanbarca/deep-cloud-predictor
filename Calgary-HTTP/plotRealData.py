import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import math
from PyEMD import EMD
import time
from datetime import datetime
from normalizer_txt import normalizer
import warnings
import psycopg2

warnings.filterwarnings("ignore")

hostname = 'localhost'
username = 'postgres'
password = 'inter2010'
database = 'cloud_load'

conn = psycopg2.connect(host=hostname, user=username, password=password, dbname=database)
cur0 = conn.cursor()
cur0.execute('delete from calgary_http_10min')
cur0.execute('delete from calgary_http_emd_10min')

conn.commit()


''' Read Raw Data '''
data = np.array(pd.read_csv('E:/ThesisNew/Thesis/Data/calgary_access_log2/calgary.csv',sep ='- -',skip_blank_lines=True))
domains=data[:,0]
timestamps=[]
ids=[]
print('total data length is ',len(domains))
for k in range(len(domains)):
    time_dates=str(data[k,1]).split('-')[0][2:-1]
    try:
        timestamps.append(datetime.strptime(normalizer(time_dates), '%d-%m-%Y:%H:%M:%S'))
    except Exception as e:
        continue
for k in range(len(timestamps)):
    ids.append(k+1)
print(len(domains),len(timestamps),len(ids))

del data

print('**************')
''' Set PWS  '''
pws=10
unique_ids_total=[]
unique_ids=[]
source=timestamps[0]
for k in range(len(timestamps)):
    td= timestamps[k]-source
    dif_min=int(round(td.total_seconds() / 60))
    #print(dif_min)
    if (dif_min)<pws:
        unique_ids.append(ids[k])
    else:
        unique_ids_total.append(unique_ids)
        unique_ids = []
        source=timestamps[k]
print(len(unique_ids_total))
print('*********')

''' extract clusters of PWS min of data '''
num_reqs=[]
sample_time=[]
for k in unique_ids_total:
    if int(len(k))>0:
        num_reqs.append(len(k))
        sample_time.append(timestamps[k[0]])
print(len(num_reqs),len(sample_time))
print('first time is ',sample_time[0])
print('last time is ',sample_time[-1])
print('------------------------------------------------------------------')

#print('Plotting histogram ...')
# plt.hist(num_reqs,bins='auto')
# plt.show()
# plt.plot(sample_time,num_reqs)
# plt.xlabel('time')
# plt.ylabel('Number of Requests')
# plt.show()

''' writing to data base'''
print('writing raw data to DB ...')
cur1=conn.cursor()
for kkk in range(len(sample_time)):
        cur1.execute('insert into calgary_http_10min values (%s,%s,%s)',(int(kkk+1),int(num_reqs[kkk]),sample_time[kkk]))
        conn.commit()


print('****** Start to EMD Analysis ...')
emd = EMD()
IMFs = emd(np.array(num_reqs,dtype=float))
print('****** EMD Analysis Completed! ')
num_of_IMF=len(IMFs)
print('total number of IMFs are',num_of_IMF)
imf_lens=[]

''' plotting IMFs'''
for imf_index in range(len(IMFs)):
        imf_lens.append(len(IMFs[imf_index]))
        for kkk in range(len(sample_time)):
            cur2 = conn.cursor()
            cur2.execute('insert into calgary_http_emd_10min values (%s,%s,%s,%s)',
                         (int(kkk + 1), int(IMFs[imf_index][kkk]), sample_time[kkk],int(imf_index+1)))
            conn.commit()
        print('IMF ' + str(imf_index + 1) + ' written to DB !!! ')
#         plt.plot(sample_time, IMFs[imf_index])
#         plt.title('imf # ' + str(imf_index + 1))
#         #plt.show()

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








