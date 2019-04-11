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



start_imf=5

def mean_absolute_percentage_error(y_true, y_pred):

    ape = []
    for k in range(len(y_true)):
        #if abs(y_true[k])!=0 and k not in z1 and k not in z2:
        if abs(y_pred[k]) > 1e-3 and abs(y_true[k]) > 1e-3:
            ape.append(abs((y_true[k] - y_pred[k]) / y_true[k]))
    plt.hist(ape, bins='auto', color='orange')
    plt.xlabel('MAPE')
    plt.ylabel('frequency')
    plt.grid()
    plt.savefig('../results/MAPE_from_imf_' + str(start_imf) + '.png', dpi=600)
    plt.pause(3)
    plt.close()
    ape=sorted(ape)
    indexes=np.where(ape<np.percentile(ape,90))[0]
    ape=[ape[k] for k in indexes]
    #print(ape)

    return np.mean(np.array(ape)) * 100

def mean_percentage_error(y_true, y_pred):

    ape = []
    for k in range(len(y_true)):
        #if abs(y_true[k])!=0  and k not in z1 and k not in z2:
        if abs(y_pred[k]) > 1e-3 and abs(y_true[k]) > 1e-3:
            ape.append(((y_true[k] - y_pred[k]) / y_true[k]))
    ape = sorted(ape)
    indexes = np.where(ape < np.percentile(ape, 90))[0]
    ape = [ape[k] for k in indexes]
    return np.mean(np.array(ape)) * 100

def median_absolute_percentage_error(y_true, y_pred):

    ape = []
    for k in range(len(y_true)):
        if abs(y_pred[k]) > 1e-3 and abs(y_true[k]) > 1e-3:
        #if abs(y_true[k])!=0  and k not in z1 and k not in z2:
            ape.append(abs((y_pred[k] - y_true[k]) / y_true[k]))
    ape = sorted(ape)
    indexes = np.where(ape < np.percentile(ape, 90))[0]
    ape = [ape[k] for k in indexes]
    return np.median(np.array(ape)) * 100

def mean_percentage_r_error(y_true, y_pred):

    ape = []
    for k in range(len(y_true)):
        if abs(y_pred[k]) > 1e-3 and abs(y_true[k]) > 1e-3:
        #if abs(y_true[k])!=0  and k not in z1 and k not in z2:
            ape.append(pow(((y_true[k] - y_pred[k]) / y_true[k]), 2))
    ape = sorted(ape)
    indexes = np.where(ape < np.percentile(ape, 90))[0]
    ape = [ape[k] for k in indexes]
    return sqrt(np.mean(np.array(ape)))


train_ts=[]
test_ts=[]
main_train_req=[]
main_test_req=[]
main_test_req_pred=[]





for i in range(start_imf,23):
    print(i,' ...')
    emd_imf=i



    cur0.execute('select ts,num_of_req,num_req_pred_svr from calgary_http_emd_10min where imf_index=%s'
                 ' and num_req_pred is null and num_req_pred_svr is not null'
                 ' order by ts',([int(emd_imf)]))
    data=np.array(cur0.fetchall())

    ts=data[:,0]
    num_req=data[:,1]
    num_req_pred=data[:,2]

    test_ts=ts
    main_test_req.append(list(num_req))
    main_test_req_pred.append(list(num_req_pred))

    print('length of test set: ',len(test_ts),len(num_req),len(num_req_pred))
    cur0.execute('select count(1) from calgary_http_emd_10min where imf_index=1 and '
                 'num_req_pred is null and num_req_pred_svr is  null')
    total=cur0.fetchall()
    total=np.array(total)[0][0]
    cur0.execute('select ts,num_of_req from calgary_http_emd_10min where imf_index=%s and '
                 'num_req_pred is null and num_req_pred_svr is null '
                 ' order by ts limit %s ', (int(emd_imf),int(total)))
    data = np.array(cur0.fetchall())
    ts_train = data[:, 0]
    num_req_train = data[:, 1]
    train_ts = ts_train
    print('length of train set: ', len(ts_train), len(num_req_train))
    main_train_req.append(list(num_req_train))

main_train_req__=np.zeros(len(train_ts))
for k in main_train_req:
    main_train_req__+=np.array(k,dtype=float)


main_test_req_=np.zeros(len(test_ts))
for k in main_test_req:
    main_test_req_+=np.array(k,dtype=float)

main_test_req_pred_=np.zeros(len(test_ts))
for k in main_test_req_pred:
    main_test_req_pred_+=np.array(k,dtype=float)

print(len(main_test_req),len(main_test_req_))
print(len(main_test_req_pred),len(main_test_req_pred_))
#
rms = sqrt(mean_squared_error(main_test_req_, main_test_req_pred_))
MPE=mean_percentage_error(main_test_req_,main_test_req_pred_)
MAPE=mean_absolute_percentage_error(main_test_req_,main_test_req_pred_)
MEAPE=median_absolute_percentage_error(main_test_req_,main_test_req_pred_)
RMSRE=mean_percentage_r_error(main_test_req_,main_test_req_pred_)

fig = plt.figure(facecolor='white',figsize=(12, 7))
ax = fig.add_subplot(211)
plt.plot(ts_train,main_train_req__,color='red',label='Real Req Train Data')
plt.plot(test_ts, main_test_req_, color='blue',alpha=0.5,
         label='Test Req')
plt.plot(test_ts,main_test_req_pred_,'-.',color='green',
         label='Prediction Req')
ax = fig.add_subplot(212)
plt.plot(ts,main_test_req_,'-',color='blue',label='Real Req',alpha=0.4)
plt.plot(ts,main_test_req_pred_,'-',color='green',alpha=0.9,
         label=('Prediction Req, MAPE = %.4f%% ,  RMSE=%.4f , MPE=%.4f%% ,\n  MEAPE=%.4f%%, RMSRE=%4f '% (MAPE,rms,MPE,MEAPE,RMSRE)))
plt.xlabel('TS for test part')
plt.ylabel('Num of Req')
plt.legend()
plt.grid()
plt.savefig('../results/main_reconstruct_from_imf_'+str(start_imf) + '.png', dpi=600)
plt.pause(7)
plt.close()

