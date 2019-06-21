import psycopg2
import numpy as np

hostname = 'localhost'
username = 'postgres'
password = 'inter2010'
database = 'load_cloud'

conn = psycopg2.connect(host=hostname, user=username, password=password, dbname=database)
cur0 =conn.cursor()

def read_data():
    total_req_smooth = []
    for imf_index in range(1, 20):
        print(imf_index)
        cur0.execute('select ts,num_of_req from saskatchewan_http_emd_30min_copy '
                     ' where imf_index={} and num_req_pred is null '
                     ' order by ts'.format(imf_index))
        data = np.array(cur0.fetchall())

        ts = data[:, 0]
        num_req = data[:, 1]

        total_req_smooth.append(list(num_req))

    main_total_req_smooth__ = np.zeros(len(ts))
    for k in total_req_smooth:
        main_total_req_smooth__ += np.array(k, dtype=float)

    return ts, main_total_req_smooth__