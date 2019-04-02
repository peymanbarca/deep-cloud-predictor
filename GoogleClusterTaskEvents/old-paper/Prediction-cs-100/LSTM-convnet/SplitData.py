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

    

    return ts,cpu_values,\
           ram_values



