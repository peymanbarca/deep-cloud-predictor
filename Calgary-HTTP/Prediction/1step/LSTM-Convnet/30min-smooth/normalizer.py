import numpy as np
from Read_Data import read_data
from matplotlib import pyplot as plt
from knn import perform_knn

def normalizer(plot=False):
    ts,num_req=read_data()


    print('-----------------------')
    from sklearn import preprocessing
    # max_abs_scaler = preprocessing.StandardScaler()
    # num_req_normalize = max_abs_scaler.fit_transform(num_req.reshape(-1, 1))
    # print('-----------------------')
    minMaxScaler = preprocessing.MinMaxScaler()
    num_req_normalize = minMaxScaler.fit_transform(num_req.reshape(-1, 1))
    print('-----------------------',len(ts),len(num_req_normalize))
    print('min = ',min(num_req_normalize),' max = ',max(num_req_normalize))
    print('-----------------------')
    # min1=min(num_req)
    # max1=max(num_req)
    # #num_req_normalize = num_req / (max1 - min1)
    # min2 = min(num_req_normalize)
    # max2 = max(num_req_normalize)
    # print('min = ', min2, ' max = ', max2)

    num_req_normalize = perform_knn(num_req_normalize)
    print('min = ',min(num_req_normalize),' max = ',max(num_req_normalize))

    if plot:
        fig=plt.figure(figsize=(8, 6))
        plt.subplot(211)
        plt.plot(ts, num_req, color='red', label='REQ-data original')
        plt.ylabel('Num of REQ original')
        plt.legend()
        plt.xlabel('Time symbol')
        plt.subplot(212)
        plt.plot(ts, num_req_normalize, color='green', label='REQ-data normalized')
        plt.ylabel('Num of REQ normalized')
        plt.legend()
        plt.xlabel('Time symbol')
        plt.pause(3)
        plt.close()

    return ts,num_req_normalize,minMaxScaler,max(num_req)





