import numpy as np
import sklearn
from sklearn import preprocessing
from SplitData import split_data
from matplotlib import pyplot as plt





def normalizer(imf_index,plot=False):
    ts, ts_train, ts_valid, ts_test, cpu_values, cpu_train, cpu_valid, cpu_test, \
            ram_values, ram_train, ram_valid, ram_test=split_data(imf_index,plot=False)


    std_cpu = np.std(cpu_values)
    std_ram = np.std(ram_values)
    mean_cpu = np.mean(cpu_values)
    mean_ram = np.mean(ram_values)
    print('means are : ',mean_ram,mean_cpu)
    print('Stds are : ', std_ram, std_cpu)
    print('*****************')


    #cpu_values_normalize = (np.array(cpu_values) - mean_cpu) / std_cpu
    max_abs_scaler = preprocessing.MaxAbsScaler()
    cpu_values_normalize = max_abs_scaler.fit_transform(cpu_values.reshape(-1, 1))
    # cpu_values_normalize = sklearn.preprocessing.normalize(cpu_values.reshape(-1, 1))
    #cpu_values_normalize=cpu_values_normalize/np.max(np.abs(cpu_values_normalize))


    #ram_values_normalize = (np.array(ram_values) - mean_ram) / std_ram
    ram_values_normalize = max_abs_scaler.fit_transform(ram_values.reshape(-1, 1))
    # ram_values_normalize = sklearn.preprocessing.normalize(ram_values.reshape(-1, 1))
    #ram_values_normalize = ram_values_normalize / np.max(np.abs(ram_values_normalize))


    # '''--------------------------  Reload Data --------------------------------------- '''
    # desired_len = 100
    # ts_reload,cpu_reloaded_normalize,ram_reloaded_normalize=\
    #         Reload_Data_RF.Reload_Data_RF(ts,cpu_values_normalize,ram_values_normalize,desired_len)
    #
    #
    # print('length of original data is ', len(cpu_values))
    # print('length of Reloaded Data is ',len(cpu_reloaded_normalize),len(ram_reloaded_normalize))
    # print('---------------------------------------------------')
    #
    # if plot:
    #     plt.subplot(2, 1, 1)
    #     plt.plot(ts, cpu_values_normalize, color='red', label='cpu-original-data')
    #     plt.ylabel('CPU Req normalized')
    #     plt.xlabel('Time symbol')
    #     plt.legend()
    #     plt.subplot(2, 1, 2)
    #     plt.plot(ts_reload, cpu_reloaded_normalize, color='blue', label='cpu-Reloaded-data')
    #     plt.ylabel('CPU Req normalized')
    #     plt.legend()
    #     plt.xlabel('Time symbol')
    #     plt.show()
    #
    #     plt.subplot(2, 1, 1)
    #     plt.plot(ts, ram_values_normalize, color='red', label='RAM-original-data')
    #     plt.ylabel('CPU Req normalized')
    #     plt.xlabel('Time symbol')
    #     plt.legend()
    #     plt.subplot(2, 1, 2)
    #     plt.plot(ts_reload, ram_reloaded_normalize, color='blue', label='RAM-Reloaded-data')
    #     plt.ylabel('CPU Req normalized')
    #     plt.legend()
    #     plt.xlabel('Time symbol')
    #     plt.show()

    # return std_cpu,std_ram,mean_cpu,mean_ram,ts,ts_reload, \
    #        cpu_values_normalize,cpu_reloaded_normalize, \
    #                 ram_values_normalize,ram_reloaded_normalize

    return std_cpu,std_ram,mean_cpu,mean_ram,ts,cpu_values_normalize,ram_values_normalize





