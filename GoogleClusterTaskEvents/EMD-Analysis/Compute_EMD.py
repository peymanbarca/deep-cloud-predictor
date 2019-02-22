import psycopg2
import numpy as np
from matplotlib import pyplot as plt
import math
from PyEMD import EMD

hostname = 'localhost'
username = 'postgres'
password = 'inter2010'
database = 'load_cloud'

conn = psycopg2.connect(host=hostname, user=username, password=password, dbname=database)

cur=conn.cursor()
cur.execute('select * from google_cpu_req_200 order by ts')
data1=np.array(cur.fetchall())
ts=data1[:,0]
cpu=data1[:,1]
cur.execute('select * from google_ram_req_200 order by ts')
data2=np.array(cur.fetchall())
ram=data2[:,1]

print('****** Start to EMD Analysis for CPU ...')
emd = EMD()
IMFs = emd(np.array(cpu, dtype=float))
print('****** EMD Analysis Completed! ')
num_of_IMF = len(IMFs)
print('total number of IMFs are', num_of_IMF)
imf_lens = []

''' plotting IMFs from CPU '''
for imf_index in range(len(IMFs)):
            imf_lens.append(len(IMFs[imf_index]))
            for kkk in range(len(ts)):
                cur2=conn.cursor()
                cur2.execute('insert into google_cpu_req_emd_200 values (%s,%s,%s,%s)',
                             (int(ts[kkk]), IMFs[imf_index][kkk],int(imf_index+1), int(imf_index + 1)))
                conn.commit()
            print('IMF ' + str(imf_index+1) + ' written to DB !!! ')


            plt.plot(ts, IMFs[imf_index])
            plt.title('imf # ' + str(imf_index + 1))
            plt.show()

print('****** Start to EMD Analysis for RAM ...')
emd = EMD()
IMFs = emd(np.array(ram, dtype=float))
print('****** EMD Analysis Completed! ')
num_of_IMF = len(IMFs)
print('total number of IMFs are', num_of_IMF)
imf_lens = []

''' plotting IMFs from RAM '''
for imf_index in range(len(IMFs)):
        imf_lens.append(len(IMFs[imf_index]))
        for kkk in range(len(ts)):
            cur2 = conn.cursor()
            cur2.execute('insert into google_ram_req_emd_200 values (%s,%s,%s,%s)',
                         (int(ts[kkk]), IMFs[imf_index][kkk], int(imf_index + 1), int(imf_index + 1)))
            conn.commit()
        print('IMF ' + str(imf_index + 1) + ' written to DB !!! ')


        plt.plot(ts, IMFs[imf_index])
        plt.title('imf # ' + str(imf_index + 1))
        plt.show()

print('length of IMFs are', imf_lens)
print('***********')
''' reconstructing main data from its IMFs'''
reconstructed_data = np.zeros([1, imf_lens[0]])
for imf_index in range(len(IMFs)):
        reconstructed_data = reconstructed_data + np.array(IMFs[imf_index])
print('RMSE between original and reconstructed signal is',
          np.sqrt(np.sum(np.square(reconstructed_data - cpu))))
plt.subplot(2, 1, 1)
plt.plot(ts, cpu, color='blue', label='original data')
plt.subplot(2, 1, 2)
plt.plot(ts, cpu, color='red', label='reconstructed data')
plt.legend()
plt.grid()
plt.show()

plt.plot(ts, list(np.transpose(np.square(reconstructed_data - cpu))), color='blue',
                 label='squared error')
plt.legend()
plt.grid()
plt.show()



