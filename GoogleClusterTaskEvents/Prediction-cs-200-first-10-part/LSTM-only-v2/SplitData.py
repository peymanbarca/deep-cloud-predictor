import numpy as np
from Read_Data import read_data
from matplotlib import pyplot as plt

def split_data(plot=False):
    ram_data,cpu_data = read_data()
    ts=ram_data[:,0]
    ram_values=ram_data[:,1]
    cpu_values=cpu_data[:,1]
    l=len(ts)
    print('length of total data are ',len(ts),len(ram_values),len(cpu_values))

    factor1=0.8  # train
    factor2 = 0.9 # valuation
    ts_train = ts[:int(factor1*l)]
    ts_valid = ts[int(factor1*l)-1:int(factor2*l)]
    ts_test = ts[int(factor2*l)-1:]
    cpu_train = cpu_values[:int(factor1 * l)]
    cpu_valid = cpu_values[int(factor1*l)-1:int(factor2*l)]
    cpu_test = cpu_values[int(factor2 * l)-1:]
    ram_train = ram_values[:int(factor1 * l)]
    ram_valid = ram_values[int(factor1 * l)-1:int(factor2 * l)]
    ram_test = ram_values[int(factor2 * l)-1:]

    if plot:
        plt.subplot(2, 1, 1)
        plt.plot(ts_train, cpu_train,color='red',label='cpu-train-data')
        plt.plot(ts_valid, cpu_valid, color='green', label='cpu-validation-data')
        plt.plot(ts_test, cpu_test,color='blue',label='cpu-test-data')
        plt.ylabel('CPU Req')
        plt.xlabel('Time symbol')
        plt.legend()
        plt.subplot(2, 1, 2)
        plt.plot(ts_train, ram_train,color='red',label='RAM-train-data')
        plt.plot(ts_valid, ram_valid, color='green', label='RAM-validation-data')
        plt.plot(ts_test, ram_test,color='blue',label='RAM-test-data')
        plt.ylabel('RAM Req')
        plt.legend()
        plt.xlabel('Time symbol')
        plt.show()

    return ts,ts_train,ts_valid,ts_test,cpu_values,cpu_train,cpu_valid,cpu_test,\
           ram_values,ram_train,ram_valid,ram_test



