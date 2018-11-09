from sklearn.ensemble import RandomForestRegressor
import numpy as np
from matplotlib import pyplot as plt


def sample_data(ts,cpu,ram,desired_len,plot=False):
    ts_reload = np.linspace(np.min(ts), np.max(ts), desired_len)

    ''' ----------- CPU ----------- '''
    rf = RandomForestRegressor(n_estimators=200, oob_score=True, random_state=0)
    rf.fit(ts.reshape(-1, 1), cpu.reshape(-1, 1).ravel())
    cpu_reloaded = []
    print('Reloading CPU data .... ')
    for k in range(len(ts_reload)):
        if k % 100 == 1:
            print(k, '...')
        cpu_reloaded.append(rf.predict(ts_reload[k].reshape(1, -1)))

    ''' ----------- RAM ----------- '''
    rf = RandomForestRegressor(n_estimators=200, oob_score=True, random_state=0)
    rf.fit(ts.reshape(-1, 1), ram.reshape(-1, 1).ravel())
    print('Reloading RAM data .... ')
    ram_reloaded = []
    for k in range(len(ts_reload)):
        if k % 100 == 1:
            print(k, '...')
        ram_reloaded.append(rf.predict(ts_reload[k].reshape(1, -1)))

    if plot:
        plt.subplot(2, 1, 1)
        plt.plot(ts, cpu, color='red', label='cpu-original-data , l= '+str(len(ts)))
        plt.ylabel('CPU Req normalized')
        plt.xlabel('Time symbol')
        plt.legend()
        plt.subplot(2, 1, 2)
        plt.plot(ts_reload, cpu_reloaded, color='blue', label='cpu-Reloaded-data , l= '+str(len(ts_reload)))
        plt.ylabel('CPU Req normalized')
        plt.legend()
        plt.xlabel('Time symbol')
        plt.show()

        plt.subplot(2, 1, 1)
        plt.plot(ts, ram, color='red', label='RAM-original-data , l= '+str(len(ts)))
        plt.ylabel('CPU Req normalized')
        plt.xlabel('Time symbol')
        plt.legend()
        plt.subplot(2, 1, 2)
        plt.plot(ts_reload, ram_reloaded, color='blue', label='RAM-Reloaded-data , l= ' +str(len(ts_reload)))
        plt.ylabel('CPU Req normalized')
        plt.legend()
        plt.xlabel('Time symbol')
        plt.show()

    return ts_reload,cpu_reloaded,ram_reloaded