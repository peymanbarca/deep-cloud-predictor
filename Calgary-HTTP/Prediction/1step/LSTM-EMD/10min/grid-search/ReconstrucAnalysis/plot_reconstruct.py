import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import psycopg2
from norm import norm_v2_single

hostname = 'localhost'
username = 'postgres'
password = 'inter2010'
database = 'load_cloud'

conn = psycopg2.connect(host=hostname, user=username, password=password, dbname=database)
cur0=conn.cursor()




def mean_absolute_percentage_error(y_true, y_pred):
    #y_true, y_pred = norm_v2_single(y_true),norm_v2_single(y_pred)
    y_true, y_pred = np.array(y_true) + np.max(y_true), np.array(y_pred) + np.max(y_pred)
    ape = []
    for k in range(len(y_true)):
        if abs(y_pred[k]) > 1e-3 and abs(y_true[k]) > 1e-3:
            ape.append(abs(     (y_true[k] - y_pred[k]) / y_true[k]          ))
    return np.mean(np.array(ape)) * 100

def mean_percentage_error(y_true, y_pred):
    #y_true, y_pred = norm_v2_single(y_true), norm_v2_single(y_pred)
    y_true, y_pred = np.array(y_true) + np.max(y_true), np.array(y_pred) + np.max(y_pred)
    ape = []
    for k in range(len(y_true)):
        if abs(y_pred[k]) > 1e-3 and abs(y_true[k]) > 1e-3:
            ape.append((     (y_true[k] - y_pred[k]) / y_true[k]          ))
    return np.mean(np.array(ape)) * 100

def median_absolute_percentage_error(y_true, y_pred):
    #y_true, y_pred = norm_v2_single(y_true), norm_v2_single(y_pred)
    y_true, y_pred = np.array(y_true)+np.max(y_true), np.array(y_pred)+np.max(y_pred)
    ape = []
    for k in range(len(y_true)):
        if abs(y_pred[k]) > 1e-3 and abs(y_true[k]) > 1e-3:
            ape.append(abs((y_pred[k] - y_true[k]) / y_true[k]))
    return np.median(np.array(ape)) * 100

for i in range(1,18):
    print(i,' ...')
    emd_imf=i
    cur0.execute('select ts,num_of_req,num_req_pred from calgary_http_emd_60min where imf_index=%s and num_req_pred is not null'
                 ' order by ts',([int(emd_imf)]))
    data=np.array(cur0.fetchall())

    ts=data[:,0]
    num_req=data[:,1]
    num_req_pred=data[:,2]

    from sklearn.metrics import mean_squared_error
    from math import sqrt

    rms = sqrt(mean_squared_error(num_req, num_req_pred))
    MPE=mean_percentage_error(num_req,num_req_pred)
    MAPE=mean_absolute_percentage_error(num_req,num_req_pred)
    MEAPE=median_absolute_percentage_error(num_req,num_req_pred)

    fig = plt.figure(facecolor='white',figsize=(12, 7))
    plt.plot(ts,num_req,color='blue',label='Real Req')
    plt.plot(ts,num_req_pred,'-.',color='green',
             label=('Prediction Req, MAPE = %.4f%% ,  RMSE=%.4f , MPE=%.4f%% ,  MEAPE=%.4f%% '% (MAPE,rms,MPE,MEAPE)))
    plt.xlabel('TS for test part')
    plt.ylabel('Num of Req')
    plt.legend()
    plt.grid()
    plt.savefig('../results/imf' + str(emd_imf) + '/Reconstruct' + str(emd_imf) + '.png', dpi=600)
    plt.pause(7)
    plt.close()

