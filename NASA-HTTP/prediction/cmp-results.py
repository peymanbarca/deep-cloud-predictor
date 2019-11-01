from matplotlib import pyplot as plt
import numpy as np

nasa_10min_lstm=[12.66]
nasa_10min_lstm_cnn=[13.06]
nasa_10min_svr=[14.15]
nasa_10min_emd_svr=[11.59]
nasa_10min_emd_lstm=[8.83]
nasa_10min_emd_gan=[10.13]
nasa_10min_our=[8.25]

nasa_5min_lstm=[14.37]
nasa_5min_lstm_cnn=[14.68]
nasa_5min_svr=[16.93]
nasa_5min_emd_svr=[9.93]
nasa_5min_emd_lstm=[10.26]
nasa_5min_emd_gan=[9.59]
nasa_5min_our=[9.00]

nasa_1min_lstm=[8.33]
nasa_1min_lstm_cnn=[9.94]
nasa_1min_svr=[9.75]
nasa_1min_emd_svr=[5.85]
nasa_1min_emd_lstm=[9.14]
nasa_1min_emd_gan=[6.68]
nasa_1min_our=[5.82]

### ----------------------------

# data to plot
n_groups = 3
Ours = (nasa_10min_our[0],nasa_5min_our[0],nasa_1min_our[0])
EMD_LSTM = (nasa_10min_emd_lstm[0],nasa_5min_emd_lstm[0],nasa_1min_emd_lstm[0])
LSTM = (nasa_10min_lstm[0],nasa_5min_lstm[0],nasa_1min_lstm[0])
EMD_GAN = (nasa_10min_emd_gan[0],nasa_5min_emd_gan[0],nasa_1min_emd_gan[0])
EMD_SVR = (nasa_10min_emd_svr[0],nasa_5min_emd_svr[0],nasa_1min_emd_svr[0])
CNN_LSTM = (nasa_10min_lstm_cnn[0],nasa_5min_lstm_cnn[0],nasa_1min_lstm_cnn[0])
SVR = (nasa_10min_svr[0],nasa_5min_svr[0],nasa_1min_svr[0])

# create plot
fig, ax = plt.subplots()
index = np.arange(3)
print(index)
bar_width = 0.1
opacity = 0.8

rects1 = plt.bar(index, Ours,bar_width,
alpha=opacity,
color='g',
label='ELG')

rects2 = plt.bar(index+bar_width ,  EMD_LSTM,bar_width,
alpha=opacity,
color='#1d2d38',
label='EMD+LSTM')

rects3 = plt.bar(index+2*bar_width , EMD_GAN,bar_width,
alpha=opacity,
color='#255e85',
label='EMD+GAN')

rects4 = plt.bar(index+3*bar_width ,  EMD_SVR,bar_width,
alpha=opacity,
color='#6593F5',
label='EMD_SVR')

rects5 = plt.bar(index+4*bar_width, CNN_LSTM,bar_width,
alpha=opacity,
color='#0F52BA',
label='CNN+LSTM')

rects6 = plt.bar(index+5*bar_width , SVR,bar_width,
alpha=opacity,
color='#000080',
label='SVR')

plt.xlabel('PWS (Minute)')
plt.ylabel('MAPE %')
plt.title('NASA HTTP Trace')
plt.xticks(index + bar_width, ('10', '5', '1'))
plt.legend()
plt.grid(True, 'major', 'y', ls='--', lw=.5, c='k', alpha=.3)
# plt.tight_layout()
plt.savefig("NASA-mape-cmp.png",dpi=500)
plt.show()