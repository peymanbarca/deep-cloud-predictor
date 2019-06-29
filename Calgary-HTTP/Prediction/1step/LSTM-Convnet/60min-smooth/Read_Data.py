import psycopg2
import numpy as np

hostname = 'localhost'
username = 'postgres'
password = 'inter2010'
database = 'load_cloud'

conn = psycopg2.connect(host=hostname, user=username, password=password, dbname=database)

def read_data():
    total_req_smooth=[]
    cur1=conn.cursor()

    for imf_index in range(1, 18):
        print(imf_index)
        cur1.execute('select ts,num_of_req from calgary_http_emd_60min_copy '
                     ' where imf_index={} and num_req_pred is null '
                     ' order by ts'.format(imf_index))
        data = np.array(cur1.fetchall())

        ts = data[:, 0]
        num_req = data[:, 1]

        total_req_smooth.append(list(num_req))

    main_total_req_smooth__ = np.zeros(len(ts))
    for k in total_req_smooth:
        main_total_req_smooth__ += np.array(k, dtype=float)

    data1=[ts,main_total_req_smooth__]
    print('data read from DB!')
    return ts,main_total_req_smooth__