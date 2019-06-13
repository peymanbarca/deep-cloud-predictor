import numpy as np
from Read_Data import read_data
from matplotlib import pyplot as plt
from knn import perform_knn
from norm import norm_v1,norm_v2

def normalizer(imf_index,ver=2,plot=False):
    data=read_data(imf_index)
    ts=data[:,0]
    num_req=data[:,1]

    print('-----------------------')

    if ver==2:
        min1, max1, min2, max2, num_req_normalize,MaxAbsScalerObj = norm_v2(num_req)
    elif ver==1:
        min1, max1, min2, max2, num_req_normalize, MaxAbsScalerObj = norm_v1(num_req)
    print('min = ', min2, ' max = ', max2)

    #num_req_normalize = perform_knn(num_req_normalize)
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

    return ts,num_req_normalize,MaxAbsScalerObj





