from matplotlib import pyplot as plt
import numpy as np

nasa_10min_lstm=[12.66]
nasa_10min_lstm_cnn=[13.06]
nasa_10min_svr=[14.15]
nasa_10min_emd_svr=[11.59]
nasa_10min_emd_lstm=[8.83]
nasa_10min_emd_gan=[11.33]
nasa_10min_our=[8.69]

nasa_5min_lstm=[14.37]
nasa_5min_lstm_cnn=[14.68]
nasa_5min_svr=[16.93]
nasa_5min_emd_svr=[9.93]
nasa_5min_emd_lstm=[10.26]
nasa_5min_emd_gan=[9.59]
nasa_5min_our=[9.73]

nasa_1min_lstm=[8.33]
nasa_1min_lstm_cnn=[9.94]
nasa_1min_svr=[9.75]
nasa_1min_emd_svr=[5.85]
nasa_1min_emd_lstm=[9.14]
nasa_1min_emd_gan=[6.99]
nasa_1min_our=[6.85]

### ----------------------------

# data to plot
n_groups = 3
Ours = (8.69,9.74,6.85)
EMD_LSTM = (8.83,10.26,9.14)
LSTM = (12.66,14.37,8.33)
EMD_GAN = (11.33,9.59,6.99)
EMD_SVR = (11.59,9.93,5.85)
CNN_LSTM = (13.06,14.68,9.94)
SVR = (14.15,16.93,9.75)

# create plot
fig, ax = plt.subplots()
index = np.arange(3)
print(index)
bar_width = 0.1
opacity = 0.8

rects1 = plt.bar(index, Ours,bar_width,
alpha=opacity,
color='g',
label='Ours')

rects2 = plt.bar(index+bar_width ,  EMD_LSTM,bar_width,
alpha=opacity,
color='b',
label='EMD_LSTM')

rects3 = plt.bar(index+2*bar_width , EMD_GAN,bar_width,
alpha=opacity,
color='cyan',
label='EMD_GAN')

rects4 = plt.bar(index+3*bar_width ,  EMD_SVR,bar_width,
alpha=opacity,
color='yellow',
label='EMD_SVR')

rects5 = plt.bar(index+4*bar_width, CNN_LSTM,bar_width,
alpha=opacity,
color='black',
label='CNN_LSTM')

rects6 = plt.bar(index+5*bar_width , SVR,bar_width,
alpha=opacity,
color='r',
label='SVR')

plt.xlabel('PWS (Minute)')
plt.ylabel('MAPE %')
plt.title('NASA HTTP Trace')
plt.xticks(index + bar_width, ('10 min', '5 min', '1 min'))
plt.legend()

# plt.tight_layout()
plt.savefig("NASA-mape-cmp.png")
plt.show()