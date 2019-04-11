import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import math
from PyEMD import EMD
import datetime
import psycopg2
''' This is version is based on task_events of google cluster data'''


def plot_4_all_cluster():

    hostname = 'localhost'
    username = 'postgres'
    password = 'inter2010'
    database = 'load_cloud'

    conn = psycopg2.connect(host=hostname, user=username, password=password, dbname=database)
    cur0=conn.cursor()
    cur0.execute('delete from google_cpu_req_100')
    #cur0.execute('delete from google_cpu_req_emd_1000')
    cur0.execute('delete from google_ram_req_100')
    #cur0.execute('delete from google_ram_req_emd_1000')
    conn.commit()

    tsReq=[]
    cpuReq=[]
    ramReq=[]
    for k in range(100):
        
        print(k,'...')
        # if k < 10:
        #     data = np.array(
        #         pd.read_csv(r'E:\ThesisNew\Thesis\Data\task_events\task_events_part-0000' + str(k) +
        #                     r'-of-00500.csv\task_events_part-0000' + str(k) + r'-of-00500.csv', usecols=[0]))
        # elif k < 100:
        #     data = np.array(
        #         pd.read_csv(r'E:\ThesisNew\Thesis\Data\task_events\task_events_part-000' + str(k) +
        #                     r'-of-00500.csv\task_events_part-000' + str(k) + r'-of-00500.csv', usecols=[0]))
        #
        # print(data.shape)
        # tsReq.append(data[:, 0])
        if k<10:
            data = np.array(
                pd.read_csv(r'E:\ThesisNew\Thesis\Data\task_events\task_events_part-0000'+str(k)+
                            r'-of-00500.csv\task_events_part-0000'+str(k)+r'-of-00500.csv',usecols=[9]))
        elif k<100:
            data = np.array(
                pd.read_csv(r'E:\ThesisNew\Thesis\Data\task_events\task_events_part-000' + str(k) +
                            r'-of-00500.csv\task_events_part-000' + str(k) + r'-of-00500.csv',usecols=[9]))

        print(data.shape)
        cpuReq.append(data[:, 0])

        del data
        if k<10:
            data = np.array(
                pd.read_csv(r'E:\ThesisNew\Thesis\Data\task_events\task_events_part-0000'+str(k)+
                            r'-of-00500.csv\task_events_part-0000'+str(k)+r'-of-00500.csv',usecols=[10]))
        elif k<100:
            data = np.array(
                pd.read_csv(r'E:\ThesisNew\Thesis\Data\task_events\task_events_part-000' + str(k) +
                            r'-of-00500.csv\task_events_part-000' + str(k) + r'-of-00500.csv',usecols=[10]))
        #timestamp.append(data[:, 0])
        print(data.shape)
        #cpuReq.append(data[:, 0])
        ramReq.append(data[:, 0])
        del data

    #tsReq = [j for i in tsReq for j in i]
    cpuReq = [j for i in cpuReq for j in i]
    ramReq = [j for i in ramReq for j in i]
    print(len(cpuReq),len(ramReq))


    cpuReq = np.array([0 if math.isnan(x) else x for x in cpuReq])  ## todo : replace it with KNNs
    ramReq = np.array([0 if math.isnan(x) else x for x in ramReq])

    print('first elemnts of cpu reqs',cpuReq[:4])
    print('first elemnts of RAM reqs',ramReq[:4])

    #unique_times = list(sorted(set(list(tsReq)))) ## sort times

    # print('first time is ',datetime.datetime.fromtimestamp(
    #     int(str(unique_times[0]))
    # ).strftime('%Y-%m-%d %H:%M:%S'))
    # print('last time is ', datetime.datetime.fromtimestamp(
    #     int(str(unique_times[-1]))
    # ).strftime('%Y-%m-%d %H:%M:%S'))
    # print('total time series length is : ',len(cpuReq), len(ramReq))
    print('-------------------------------------')



    ''' set prediction window size'''
    chunk_size=100
    print('total chunk timestamps are ',str(int(len(cpuReq)/chunk_size)))
    total_chunks=int(len(cpuReq)/chunk_size)

    timeCpuReq = []
    timeRamReq = []
    time_stamp_symbol=[]
    time_stamp_real=[]


    for k in range(total_chunks):
        time_stamp_symbol.append(k)


        print('Timestamp chunk of ',str(k),' are collecting .... ')
        #time_stamp_real.append(unique_times[k*chunk_size])

        timeCpuReq.append(np.sum(cpuReq[k*chunk_size:(k+1)*chunk_size]))

        timeRamReq.append(np.sum(ramReq[k*chunk_size:(k+1)*chunk_size]))
    #
    #
    del cpuReq,ramReq

    print(len(timeCpuReq), len(timeRamReq))

    plt.subplot(2, 1, 1)
    plt.plot(time_stamp_symbol, timeCpuReq)
    plt.xlabel('Time')
    plt.ylabel('CPU Req')
    plt.subplot(2, 1, 2)
    plt.plot(time_stamp_symbol, timeRamReq,color='red')
    plt.xlabel('Time')
    plt.ylabel('RAM Req')
    plt.show()
    #

    ''' writing to data base'''
    print('writing raw data to DB ...')
    cur1=conn.cursor()
    for kkk in range(len(time_stamp_symbol)):
        cur1.execute('insert into google_ram_req_100 values (%s,%s)',(int(time_stamp_symbol[kkk]),timeRamReq[kkk]))
        conn.commit()
        cur1.execute('insert into google_cpu_req_100 values (%s,%s)', (int(time_stamp_symbol[kkk]), timeCpuReq[kkk]))
        conn.commit()

    # print('****** Start to 60min Analysis for CPU ...')
    # emd = 60min()
    # IMFs = emd(np.array(timeCpuReq, dtype=float))
    # print('****** 60min Analysis Completed! ')
    # num_of_IMF = len(IMFs)
    # print('total number of IMFs are', num_of_IMF)
    # imf_lens = []
    #
    # ''' plotting IMFs from CPU '''
    # for imf_index in range(len(IMFs)):
    #         imf_lens.append(len(IMFs[imf_index]))
    #         for kkk in range(len(time_stamp_symbol)):
    #             cur2=conn.cursor()
    #             cur2.execute('insert into google_cpu_req_emd_200 values (%s,%s,%s,%s)',
    #                          (int(time_stamp_symbol[kkk]), IMFs[imf_index][kkk],int(imf_index+1), int(imf_index + 1)))
    #             conn.commit()
    #         print('IMF ' + str(imf_index+1) + ' written to DB !!! ')
    #
    #
    #         #plt.plot(time_stamp_symbol, IMFs[imf_index])
    #         #plt.title('imf # ' + str(imf_index + 1))
    #         #plt.show()
    #
    # print('****** Start to 60min Analysis for RAM ...')
    # emd = 60min()
    # IMFs = emd(np.array(timeRamReq, dtype=float))
    # print('****** 60min Analysis Completed! ')
    # num_of_IMF = len(IMFs)
    # print('total number of IMFs are', num_of_IMF)
    # imf_lens = []
    #
    # ''' plotting IMFs from RAM '''
    # for imf_index in range(len(IMFs)):
    #     imf_lens.append(len(IMFs[imf_index]))
    #     for kkk in range(len(time_stamp_symbol)):
    #         cur2 = conn.cursor()
    #         cur2.execute('insert into google_ram_req_emd_200 values (%s,%s,%s,%s)',
    #                      (int(time_stamp_symbol[kkk]), IMFs[imf_index][kkk], int(imf_index + 1), int(imf_index + 1)))
    #         conn.commit()
    #     print('IMF ' + str(imf_index + 1) + ' written to DB !!! ')
    #
    #
    #     # plt.plot(time_stamp_symbol, IMFs[imf_index])
    #     # plt.title('imf # ' + str(imf_index + 1))
    #     # plt.show()
    #
    # print('length of IMFs are', imf_lens)
    # print('***********')
    # ''' reconstructing main data from its IMFs'''
    # reconstructed_data = np.zeros([1, imf_lens[0]])
    # for imf_index in range(len(IMFs)):
    #     reconstructed_data = reconstructed_data + np.array(IMFs[imf_index])
    # print('RMSE between original and reconstructed signal is',
    #       np.sqrt(np.sum(np.square(reconstructed_data - timeCpuReq))))
    # plt.subplot(2, 1, 1)
    # plt.plot(time_stamp_symbol, timeCpuReq, color='blue', label='original data')
    # plt.subplot(2, 1, 2)
    # plt.plot(time_stamp_symbol, timeCpuReq, color='red', label='reconstructed data')
    # plt.legend()
    # plt.grid()
    # plt.show()
    #
    # plt.plot(time_stamp_symbol, list(np.transpose(np.square(reconstructed_data - timeCpuReq))), color='blue',
    #          label='squared error')
    # plt.legend()
    # plt.grid()
    # plt.show()





if __name__=="__main__":


    plot_4_all_cluster()

