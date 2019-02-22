import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import math
from PyEMD import EMD
import datetime
import psycopg2
''' This is version is based on task_events of google cluster data'''

def plot_for_single_jobId(jobId):
    data=np.array(pd.read_csv('E:/Thesis/DataSet/task_events_part-00000-of-00500.csv/task_events_part-00000-of-00500.csv'))

    timestamp=data[:,2]
    jobIds=data[:,4]
    cpuReq=data[:,9]
    ramReq=data[:,10]
    cpuReq = np.array([0 if math.isnan(x) else x for x in cpuReq])
    ramReq = np.array([0 if math.isnan(x) else x for x in ramReq])

    unique_times=list(sorted(set(list(timestamp))))
    print(len(cpuReq),len(ramReq),len(unique_times))



    timestamp=timestamp[np.where(jobIds==jobId)[0]]
    cpuReq=cpuReq[np.where(jobIds==jobId)[0]]
    ramReq=ramReq[np.where(jobIds==jobId)[0]]
    print(len(cpuReq),len(ramReq))

    timeCpuReq=[]
    timeRamReq=[]
    for time in unique_times:
        ids=np.where(timestamp==time)[0]
        if ids!=[]:
            timeCpuReq.append(np.sum(cpuReq[ids]))
            timeRamReq.append(np.sum(ramReq[ids]))
        else:
            timeCpuReq.append(0)
            timeRamReq.append(0)

    #
    print(len(timeCpuReq),len(timeRamReq))
    plt.subplot(2, 1, 1)
    plt.plot(unique_times,timeCpuReq)
    plt.subplot(2, 1, 2)
    plt.plot(unique_times,timeRamReq)
    plt.show()

def plot_4_all_cluster():

    hostname = 'localhost'
    username = 'postgres'
    password = 'inter2010'
    database = 'load_cloud'

    conn = psycopg2.connect(host=hostname, user=username, password=password, dbname=database)
    cur0=conn.cursor()
    cur0.execute('delete from google_cpu_req_200')
    cur0.execute('delete from google_cpu_req_emd_200')
    cur0.execute('delete from google_ram_req_200')
    cur0.execute('delete from google_ram_req_emd_200')
    conn.commit()


    timestamp=[]
    cpuReq=[]
    ramReq=[]
    for k in [0,1,2,3,4,5,6,7,8,9,10]:
        print(k,'...')
        if k<10:
            data = np.array(
                pd.read_csv(r'E:\ThesisNew\Thesis\Data\task_events\task_events_part-0000'+str(k)+
                            r'-of-00500.csv\task_events_part-0000'+str(k)+r'-of-00500.csv'))
        elif k<100:
            data = np.array(
                pd.read_csv(r'E:\ThesisNew\Thesis\Data\task_events\task_events_part-000' + str(k) +
                            r'-of-00500.csv\task_events_part-000' + str(k) + r'-of-00500.csv'))
        timestamp.append(data[:, 0])
        cpuReq.append(data[:, 9])
        ramReq.append(data[:, 10])
        del data

    timestamp = [j for i in timestamp for j in i]
    cpuReq = [j for i in cpuReq for j in i]
    ramReq = [j for i in ramReq for j in i]
    print(len(cpuReq),len(ramReq),len(timestamp))

    cpuReq = np.array([0 if math.isnan(x) else x for x in cpuReq])  ## todo : replace it with KNNs
    ramReq = np.array([0 if math.isnan(x) else x for x in ramReq])

    unique_times = list(sorted(set(list(timestamp)))) ## sort times
    print(unique_times[:4],unique_times[-4:-1])

    # print('first time is ',datetime.datetime.fromtimestamp(
    #     int(str(unique_times[0]))
    # ).strftime('%Y-%m-%d %H:%M:%S'))
    # print('last time is ', datetime.datetime.fromtimestamp(
    #     int(str(unique_times[-1]))
    # ).strftime('%Y-%m-%d %H:%M:%S'))
    print('total time series length is : ',len(cpuReq), len(ramReq), len(unique_times))



    ''' set prediction window size'''
    chunk_size=200
    print('total chunk timestamps are ',str(int(len(cpuReq)/chunk_size)))
    chunk_ts = [unique_times[x:x + chunk_size] for x in range(0, len(unique_times), chunk_size)]
    timeCpuReq = []
    timeRamReq = []
    time_stamp_symbol=[]

    unique_times=np.array(unique_times)
    k=0
    for time in chunk_ts:
        k+=1
        time_stamp_symbol.append(k)
        #print('len of this chunk is',len(time),', first step is ',time[0])
        ids1 = list(np.where(time[0] <=unique_times )[0])
        ids2 = list(np.where( unique_times <= time[-1])[0])
        ids = list(set(ids1).intersection(ids2))
        print('Timestamp chunk of ',str(k),' are collecting .... ')
        #print('num of requests in this chunk timestamp is ',len(ids))
        timeCpuReq.append(np.sum(cpuReq[ids]))
        timeRamReq.append(np.sum(ramReq[ids]))
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
        cur1.execute('insert into google_ram_req_200 values (%s,%s)',(int(time_stamp_symbol[kkk]),timeRamReq[kkk]))
        conn.commit()
        cur1.execute('insert into google_cpu_req_200 values (%s,%s)', (int(time_stamp_symbol[kkk]), timeCpuReq[kkk]))
        conn.commit()

    print('****** Start to EMD Analysis for CPU ...')
    emd = EMD()
    IMFs = emd(np.array(timeCpuReq, dtype=float))
    print('****** EMD Analysis Completed! ')
    num_of_IMF = len(IMFs)
    print('total number of IMFs are', num_of_IMF)
    imf_lens = []

    ''' plotting IMFs from CPU '''
    for imf_index in range(len(IMFs)):
            imf_lens.append(len(IMFs[imf_index]))
            for kkk in range(len(time_stamp_symbol)):
                cur2=conn.cursor()
                cur2.execute('insert into google_cpu_req_emd_200 values (%s,%s,%s,%s)',
                             (int(time_stamp_symbol[kkk]), IMFs[imf_index][kkk],int(imf_index+1), int(imf_index + 1)))
                conn.commit()
            print('IMF ' + str(imf_index+1) + ' written to DB !!! ')


            #plt.plot(time_stamp_symbol, IMFs[imf_index])
            #plt.title('imf # ' + str(imf_index + 1))
            #plt.show()

    print('****** Start to EMD Analysis for RAM ...')
    emd = EMD()
    IMFs = emd(np.array(timeRamReq, dtype=float))
    print('****** EMD Analysis Completed! ')
    num_of_IMF = len(IMFs)
    print('total number of IMFs are', num_of_IMF)
    imf_lens = []

    ''' plotting IMFs from RAM '''
    for imf_index in range(len(IMFs)):
        imf_lens.append(len(IMFs[imf_index]))
        for kkk in range(len(time_stamp_symbol)):
            cur2 = conn.cursor()
            cur2.execute('insert into google_ram_req_emd_200 values (%s,%s,%s,%s)',
                         (int(time_stamp_symbol[kkk]), IMFs[imf_index][kkk], int(imf_index + 1), int(imf_index + 1)))
            conn.commit()
        print('IMF ' + str(imf_index + 1) + ' written to DB !!! ')


        # plt.plot(time_stamp_symbol, IMFs[imf_index])
        # plt.title('imf # ' + str(imf_index + 1))
        # plt.show()

    print('length of IMFs are', imf_lens)
    print('***********')
    ''' reconstructing main data from its IMFs'''
    reconstructed_data = np.zeros([1, imf_lens[0]])
    for imf_index in range(len(IMFs)):
        reconstructed_data = reconstructed_data + np.array(IMFs[imf_index])
    print('RMSE between original and reconstructed signal is',
          np.sqrt(np.sum(np.square(reconstructed_data - timeCpuReq))))
    plt.subplot(2, 1, 1)
    plt.plot(time_stamp_symbol, timeCpuReq, color='blue', label='original data')
    plt.subplot(2, 1, 2)
    plt.plot(time_stamp_symbol, timeCpuReq, color='red', label='reconstructed data')
    plt.legend()
    plt.grid()
    plt.show()

    plt.plot(time_stamp_symbol, list(np.transpose(np.square(reconstructed_data - timeCpuReq))), color='blue',
             label='squared error')
    plt.legend()
    plt.grid()
    plt.show()





if __name__=="__main__":
    jobId = 317495559
    #plot_for_single_jobId(jobId)

    plot_4_all_cluster()

