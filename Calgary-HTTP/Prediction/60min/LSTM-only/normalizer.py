import numpy as np

from Read_Data import read_data
from matplotlib import pyplot as plt

def normalizer(plot=False):
    data=read_data()
    ts=np.array(data[:,0])
    num_req=data[:,1]


    from sklearn import preprocessing
    # max_abs_scaler = preprocessing.StandardScaler()
    # req_values_normalize = max_abs_scaler.fit_transform(num_req.reshape(-1, 1))

    MinMaxScaler = preprocessing.MinMaxScaler()
    req_values_normalize = MinMaxScaler.fit_transform(num_req.reshape(-1, 1))






    if plot:
        plt.subplot(211)
        plt.plot(ts, num_req, color='red', label='REQ-data')
        plt.ylabel('Num of REQ normalized')
        plt.legend()
        plt.xlabel('Time symbol')
        plt.subplot(212)
        plt.plot(ts, req_values_normalize, color='green', label='REQ-data Normalized')
        plt.ylabel('Num of REQ normalized')
        plt.legend()
        plt.xlabel('Time symbol')
        plt.show()

    return ts,req_values_normalize





