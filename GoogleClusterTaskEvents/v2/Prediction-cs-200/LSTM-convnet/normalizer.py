import numpy as np

from SplitData import split_data
from matplotlib import pyplot as plt
import time

def normalizer(plot=False):
    ts, cpu_values,\
            ram_values=split_data(plot=False)

    std_cpu = np.std(cpu_values)
    std_ram = np.std(ram_values)
    mean_cpu = np.mean(cpu_values)
    mean_ram = np.mean(ram_values)


    #cpu_values_normalize = (np.array(cpu_values) - mean_cpu) / std_cpu
    #ram_values_normalize = (np.array(ram_values) - mean_ram) / std_ram

    # cpu_max=np.max(cpu_values)
    # cpu_min=np.min(cpu_values)
    # ram_max=np.max(ram_values)
    # ram_min=np.min(ram_values)
    # print(cpu_max, cpu_min)
    # print(ram_max, ram_min)
    # print('----------------------')
    # #
    # cpu_values_normalize = cpu_values / (cpu_max - cpu_min)
    # ram_values_normalize = ram_values / (ram_max - ram_min)





    from sklearn import preprocessing

    min_max_scaler = preprocessing.MinMaxScaler()
    cpu_values_normalize = min_max_scaler.fit_transform(cpu_values.reshape(-1,1))
    ram_values_normalize = min_max_scaler.fit_transform(ram_values.reshape(-1, 1))

    from knn import perform_knn
    cpu_values_normalize = perform_knn(cpu_values_normalize)
    ram_values_normalize = perform_knn(ram_values_normalize)

    cpu_max = np.max(cpu_values_normalize)
    cpu_min = np.min(cpu_values_normalize)
    ram_max = np.max(ram_values_normalize)
    ram_min = np.min(ram_values_normalize)
    print(cpu_max, cpu_min)
    print(ram_max, ram_min)
    print('----------------------')



    # mean_cpu_n=np.mean(cpu_values_normalize)
    # mean_ram_n = np.mean(ram_values_normalize)
    # std_cpu_n = np.std(cpu_values_normalize)
    # std_ram_n = np.std(ram_values_normalize)
    # cpu_values_normalize = (np.array(cpu_values_normalize) - mean_cpu_n) / std_cpu_n
    # ram_values_normalize = (np.array(ram_values_normalize) - mean_ram_n) /std_ram_n

    if plot:
        plt.subplot(2, 1, 1)
        plt.plot(ts, cpu_values_normalize, color='cyan', label='CPU')
        plt.ylabel('CPU Req normalized')
        plt.xlabel('Time symbol')
        plt.legend()
        plt.subplot(2, 1, 2)
        plt.plot(ts, ram_values_normalize, color='green', label='RAM')
        plt.ylabel('RAM Req normalized')
        plt.legend()
        plt.xlabel('Time symbol')
        plt.pause(3)
        plt.close()

    return std_cpu,std_ram,mean_cpu,mean_ram,ts, \
           cpu_values_normalize,ram_values_normalize,min_max_scaler






