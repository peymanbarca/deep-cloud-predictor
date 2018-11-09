import numpy as np

from SplitData import split_data
from matplotlib import pyplot as plt

def normalizer(plot=False):
    ts, ts_train, ts_valid, ts_test, cpu_values, cpu_train, cpu_valid, cpu_test, \
            ram_values, ram_train, ram_valid, ram_test=split_data(plot=False)

    std_cpu = np.std(cpu_values)
    std_ram = np.std(ram_values)
    mean_cpu = np.mean(cpu_values)
    mean_ram = np.mean(ram_values)

    # std_cpu1 = np.std(cpu_train)
    # std_cpu2 = np.std(cpu_test)
    # std_ram1 = np.std(ram_train)
    # std_ram2 = np.std(ram_test)
    # mean_cpu1 = np.mean(cpu_train)
    # mean_cpu2 = np.mean(cpu_test)
    # mean_ram1 = np.mean(ram_train)
    # mean_ram2 = np.mean(ram_test)
    #
    # cpu_values_normalize = (np.array(cpu_values)-mean_cpu)/std_cpu
    # ram_values_normalize = (np.array(ram_values) - mean_ram) / std_ram

    cpu_values_normalize = (np.array(cpu_values) - mean_cpu) / std_cpu
    cpu_values_normalize_train = (np.array(cpu_train) - mean_cpu) / std_cpu
    cpu_values_normalize_valid = (np.array(cpu_valid) - mean_cpu) / std_cpu
    cpu_values_normalize_test = (np.array(cpu_test) - mean_cpu) / std_cpu

    ram_values_normalize = (np.array(ram_values) - mean_ram) / std_ram
    ram_values_normalize_test = (np.array(ram_test) - mean_ram) / std_ram
    ram_values_normalize_valid = (np.array(ram_valid) - mean_ram) / std_ram
    ram_values_normalize_train = (np.array(ram_train) - mean_ram) / std_ram

    if plot:
        plt.subplot(2, 1, 1)
        plt.plot(ts_train, cpu_values_normalize_train, color='red', label='cpu-train-data')
        plt.plot(ts_valid, cpu_values_normalize_valid, color='green', label='cpu-validation-data')
        plt.plot(ts_test, cpu_values_normalize_test, color='blue', label='cpu-test-data')
        plt.ylabel('CPU Req normalized')
        plt.xlabel('Time symbol')
        plt.legend()
        plt.subplot(2, 1, 2)
        plt.plot(ts_train, ram_values_normalize_train, color='red', label='RAM-train-data')
        plt.plot(ts_valid, ram_values_normalize_valid, color='green', label='RAM-validation-data')
        plt.plot(ts_test, ram_values_normalize_test, color='blue', label='RAM-test-data')
        plt.ylabel('RAM Req normalized')
        plt.legend()
        plt.xlabel('Time symbol')
        plt.show()

    return std_cpu,std_ram,mean_cpu,mean_ram,ts,ts_train,ts_valid,ts_test, \
           cpu_values_normalize,cpu_values_normalize_train,cpu_values_normalize_valid,cpu_values_normalize_test, \
                    ram_values_normalize,ram_values_normalize_train,ram_values_normalize_valid,ram_values_normalize_test






