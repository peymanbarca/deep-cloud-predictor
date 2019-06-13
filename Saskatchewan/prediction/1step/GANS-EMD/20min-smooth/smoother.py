import psycopg2
import numpy as np

hostname = 'localhost'
username = 'postgres'
password = 'inter2010'
database = 'load_cloud'

conn = psycopg2.connect(host=hostname, user=username, password=password, dbname=database)
cur1=conn.cursor()

def read_data(imf_num):

    cur1.execute('select ts,num_of_req from saskatchewan_http_emd_20min where imf_index=%s'
                 ' order by ts',([int(imf_num)]))
    data=np.array(cur1.fetchall())
    print('data read from DB!')
    return data


def smooth(x,size=5):
    from pandas import Series
    from matplotlib import pyplot
    import pandas as pd

    data=x
    x=pd.DataFrame(data=data    # values
              )
    rolling = x.rolling(window=size)
    rolling_mean = rolling.mean()
    #print(rolling_mean.head(10))
    # plot original and transformed dataset
    x.plot()
    rolling_mean.plot(color='red')
    rolling_mean=rolling_mean.dropna()
    pyplot.pause(3)
    pyplot.close()
    return np.array(rolling_mean)



from numpy import *
from matplotlib import pyplot as plt


def smooth_demo(imf_index):
    data = read_data(imf_index)
    ts = data[:, 0]
    num_req = data[:, 1]

    size=10
    num_req_smooth = smooth(num_req,size=size)
    num_req_smooth= np.concatenate([np.zeros(shape=(size-1,1)), num_req_smooth])

    print('----------')
    print(len(ts),len(num_req_smooth),len(num_req))
    plt.subplot(211)
    plt.plot(ts,num_req,label='real data')
    plt.subplot(212)
    plt.plot(ts,num_req_smooth,'green',label='smooth data ')
    plt.legend()
    plt.show()
    #
    for k in range(len(ts)):
        print(k)
        cur1.execute('update saskatchewan_http_emd_20min_copy set num_of_req=%s where imf_index=%s'
                    ' and ts=%s', \
                    (int(num_req_smooth[k]),int(imf_index), int(ts[k])))
        conn.commit()


if __name__ == '__main__':
    smooth_demo(3)