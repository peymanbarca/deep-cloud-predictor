import load_data
import pandas as pd
import plotly
import sample_data
import numpy as np

imf_number=5


ram_data,cpu_data=load_data.read_data(imf_number)
ram=pd.DataFrame({'time':ram_data[:,0],'ram_values':ram_data[:,1]})
cpu=pd.DataFrame({'time':cpu_data[:,0],'cpu_values':cpu_data[:,1]})
#print(cpu.head())

''' got sampled data '''
desired_len=300
ts_reload,cpu_reloaded,ram_reloaded =\
            sample_data.sample_data(np.array(cpu[['time']]),np.array(cpu[['cpu_values']]),
                                    np.array(ram[['ram_values']]),desired_len,plot=True)

print(len(ts_reload),len(ram_reloaded),len(cpu_reloaded))
ram_reloaded=pd.DataFrame({'time':ts_reload,'ram_values':ram_reloaded})
cpu_reloaded=pd.DataFrame({'time':ts_reload,'cpu_values':cpu_reloaded})
cpu=cpu_reloaded
ram=ram_reloaded


l=len(cpu[['time']])
print('length of sampled data is ',l)
factor=int(0.85*l)

train_cpu=cpu.loc[:factor]
test_cpu=cpu.loc[factor:]
print('train length is ',len(train_cpu[['cpu_values']]),'test length is ',len(test_cpu[['cpu_values']]))
print('--------------------------------------')

import matplotlib.pyplot as plt
# plt.plot(train_cpu[['time']],train_cpu[['cpu_values']],color='red',label='training data')
# plt.plot(test_cpu[['time']],test_cpu[['cpu_values']],color='blue',label='test data')
# plt.legend()
# plt.show()

from pyramid.arima import auto_arima

print('Running stepwise model to find best AIC ...')
stepwise_model = auto_arima(train_cpu[['cpu_values']], start_p=1, start_q=1,
                           max_p=2, max_q=2, m=12,
                           start_P=0, seasonal=True,
                           d=1, D=1, trace=True,
                           error_action='ignore',
                           suppress_warnings=True,
                           stepwise=True)
print(stepwise_model.aic())



stepwise_model.fit(train_cpu[['cpu_values']])
future_forecast = stepwise_model.predict(n_periods=len(test_cpu[['cpu_values']]))
plt.plot(train_cpu[['time']],train_cpu[['cpu_values']],color='red',label='training data')
plt.plot(test_cpu[['time']],test_cpu[['cpu_values']],color='blue',label='test data')
plt.plot(test_cpu[['time']],list(future_forecast),color='green',label='prediction')
plt.legend()
plt.show()



