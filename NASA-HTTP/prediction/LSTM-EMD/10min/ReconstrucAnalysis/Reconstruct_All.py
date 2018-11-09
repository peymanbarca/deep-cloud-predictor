import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import psycopg2
from norm import norm_v2_single
from sklearn.metrics import mean_squared_error
from math import sqrt
import itertools

hostname = 'localhost'
username = 'postgres'
password = 'inter2010'
database = 'cloud_load'

conn = psycopg2.connect(host=hostname, user=username, password=password, dbname=database)
cur0=conn.cursor()




def mean_absolute_percentage_error(y_true, y_pred):
    #y_true, y_pred = norm_v2_single(y_true),norm_v2_single(y_pred)
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    ape = []
    for k in range(len(y_true)):
        if abs(y_pred[k]) > 1e-3 and abs(y_true[k]) > 1e-3:
            ape.append(abs(     (y_true[k] - y_pred[k]) / y_true[k]          ))
    return np.mean(np.array(ape)) * 100

def mean_percentage_error(y_true, y_pred):
    #y_true, y_pred = norm_v2_single(y_true), norm_v2_single(y_pred)
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    ape = []
    for k in range(len(y_true)):
        if abs(y_pred[k]) > 1e-3 and abs(y_true[k]) > 1e-3:
            ape.append((     (y_true[k] - y_pred[k]) / y_true[k]          ))
    return np.mean(np.array(ape)) * 100

def median_absolute_percentage_error(y_true, y_pred):
    #y_true, y_pred = norm_v2_single(y_true), norm_v2_single(y_pred)
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    ape = []
    for k in range(len(y_true)):
        if abs(y_pred[k]) > 1e-3 and abs(y_true[k]) > 1e-3:
            ape.append(abs((y_pred[k] - y_true[k]) / y_true[k]))
    return np.median(np.array(ape)) * 100


train_ts=[]
test_ts=[]
main_train_req=[]
main_test_req=[]
main_test_req_pred=[]





for i in range(3,16):
    print(i,' ...')
    emd_imf=i



    cur0.execute('select ts,num_of_req,num_req_pred from nasa_http_emd_10min where imf_index=%s and num_req_pred is not null'
                 ' order by ts',([int(emd_imf)]))
    data=np.array(cur0.fetchall())

    ts=data[:,0]
    num_req=data[:,1]
    num_req_pred=data[:,2]

    test_ts=ts
    main_test_req.append(list(num_req))
    main_test_req_pred.append(list(num_req_pred))

    cur0.execute('select count(1) from nasa_http_emd_10min where imf_index=1 and num_req_pred is null')
    total=cur0.fetchall()
    total=np.array(total)[0][0]
    cur0.execute('select ts,num_of_req from nasa_http_emd_10min where imf_index=%s and num_req_pred is  null '
                 ' order by ts limit %s', (int(emd_imf),int(total-len(test_ts))))
    data = np.array(cur0.fetchall())
    ts_train = data[:, 0]
    num_req_train = data[:, 1]
    train_ts = ts_train
    main_train_req.append(list(num_req_train))

main_train_req__=np.zeros(len(train_ts))
for k in main_train_req:
    main_train_req__+=np.array(k)


main_test_req_=np.zeros(len(test_ts))
for k in main_test_req:
    main_test_req_+=np.array(k)

main_test_req_pred_=np.zeros(len(test_ts))
for k in main_test_req_pred:
    main_test_req_pred_+=np.array(k)

print(len(main_test_req),len(main_test_req_))
print(len(main_test_req_pred),len(main_test_req_pred_))
#
rms = sqrt(mean_squared_error(main_test_req_, main_test_req_pred_))
MPE=mean_percentage_error(main_test_req_,main_test_req_pred_)
MAPE=mean_absolute_percentage_error(main_test_req_,main_test_req_pred_)
MEAPE=median_absolute_percentage_error(main_test_req_,main_test_req_pred_)

fig = plt.figure(facecolor='white',figsize=(12, 7))
ax = fig.add_subplot(211)
plt.plot(ts_train,main_train_req__,'-',color='red',label='Real Req Train Data')
plt.plot(test_ts, main_test_req_, '-', color='blue',alpha=0.7,
         label='Test Req')
plt.plot(test_ts,main_test_req_pred_,'-',color='green',
         label='Prediction Req')
ax = fig.add_subplot(212)
plt.plot(ts,main_test_req_,'-',color='blue',label='Real Req')
plt.plot(ts,main_test_req_pred_,'-',color='green',alpha=0.7,
         label=('Prediction Req, MAPE = %.4f ,  RMSE=%.4f , MPE=%.4f ,  MEAPE=%.4f '% (MAPE,rms,MPE,MEAPE)))
plt.xlabel('TS for test part')
plt.ylabel('Num of Req')
plt.legend()
plt.grid()
plt.savefig('../results/main_reconstruct_from_imf_3' + '.png', dpi=600)
plt.pause(7)
plt.close()

