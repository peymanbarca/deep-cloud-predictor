from sklearn.ensemble import RandomForestRegressor
import numpy as np

def Reload_Data_RF(ts,cpu_values_normalize,ram_values_normalize,desired_len):
    ts_reload = np.linspace(np.min(ts), np.max(ts), desired_len)

    ''' ----------- CPU ----------- '''
    rf = RandomForestRegressor(n_estimators=500, oob_score=True, random_state=0)
    rf.fit(ts.reshape(-1, 1), cpu_values_normalize.reshape(-1, 1))
    cpu_reloaded_normalize = []
    print('Reloading CPU data .... ')
    for k in range(len(ts_reload)):
        if k % 100 == 1:
            print(k, '...')
        cpu_reloaded_normalize.append(rf.predict(ts_reload[k].reshape(1, -1)))

    ''' ----------- RAM ----------- '''
    rf = RandomForestRegressor(n_estimators=500, oob_score=True, random_state=0)
    rf.fit(ts.reshape(-1, 1), ram_values_normalize.reshape(-1, 1))
    print('Reloading RAM data .... ')
    ram_reloaded_normalize = []
    for k in range(len(ts_reload)):
        if k % 100 == 1:
            print(k, '...')
        ram_reloaded_normalize.append(rf.predict(ts_reload[k].reshape(1, -1)))

    return ts_reload,cpu_reloaded_normalize,ram_reloaded_normalize