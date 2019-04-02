import numpy as np

from SplitData import split_data
from matplotlib import pyplot as plt
from sklearn.ensemble import RandomForestRegressor
import Reload_Data_RF



def normalizer(imf_index,plot=False):
    ts, ts_train, ts_valid, ts_test, cpu_values, cpu_train, cpu_valid, cpu_test, \
            ram_values, ram_train, ram_valid, ram_test=split_data(imf_index,plot=False)


    std_cpu = np.std(cpu_values)
    std_ram = np.std(ram_values)
    mean_cpu = np.mean(cpu_values)
    mean_ram = np.mean(ram_values)


    cpu_values_normalize = (np.array(cpu_values) - mean_cpu) / std_cpu
    cpu_values_normalize_train = (np.array(cpu_train) - mean_cpu) / std_cpu
    cpu_values_normalize_valid = (np.array(cpu_valid) - mean_cpu) / std_cpu
    cpu_values_normalize_test = (np.array(cpu_test) - mean_cpu) / std_cpu

    ram_values_normalize = (np.array(ram_values) - mean_ram) / std_ram
    ram_values_normalize_test = (np.array(ram_test) - mean_ram) / std_ram
    ram_values_normalize_valid = (np.array(ram_valid) - mean_ram) / std_ram
    ram_values_normalize_train = (np.array(ram_train) - mean_ram) / std_ram

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





