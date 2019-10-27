from matplotlib import pyplot as plt
import numpy as np

sasc_30min_lstm=[14.69]
sasc_30min_lstm_cnn=[7.79]
sasc_30min_svr=[11.02]
sasc_30min_emd_svr=[15.78]
sasc_30min_emd_lstm=[9.56]
sasc_30min_emd_gan=[11.73]
sasc_30min_our=[9.09]

sasc_20min_lstm=[13.68]
sasc_20min_lstm_cnn=[8.55]
sasc_20min_svr=[11.06]
sasc_20min_emd_svr=[14.30]
sasc_20min_emd_lstm=[9.69]
sasc_20min_emd_gan=[12.20]
sasc_20min_our=[9.33]

sasc_10min_lstm=[11.19]
sasc_10min_lstm_cnn=[8.36]
sasc_10min_svr=[16.71]
sasc_10min_emd_svr=[11.20]
sasc_10min_emd_lstm=[6.62]
sasc_10min_emd_gan=[9.40]
sasc_10min_our=[6.58]

# -----------------------#
# data to plot
n_groups = 3
Ours = (9.09,9.33,6.58)
EMD_LSTM = (9.56,9.69,6.62)
LSTM = (14.69,13.68,11.19)
EMD_GAN = (11.73,12.20,9.40)
EMD_SVR = (15.78,14.30,11.20)
CNN_LSTM = (7.79,8.55,8.36)
SVR = (11.02,11.06,16.71)

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
label='EMD_LSTM')

rects3 = plt.bar(index+2*bar_width , EMD_GAN,bar_width,
alpha=opacity,
color='#255e85',
label='EMD_GAN')

rects4 = plt.bar(index+3*bar_width ,  EMD_SVR,bar_width,
alpha=opacity,
color='#6593F5',
label='EMD_SVR')

rects5 = plt.bar(index+4*bar_width, CNN_LSTM,bar_width,
alpha=opacity,
color='#0F52BA',
label='CNN_LSTM')

rects6 = plt.bar(index+5*bar_width , SVR,bar_width,
alpha=opacity,
color='#000080',
label='SVR')

plt.xlabel('PWS (Minute)')
plt.ylabel('MAPE %')
plt.title('Saskatchewan HTTP Trace')
plt.xticks(index + bar_width, ('30 min', '20 min', '10 min'))
plt.legend()
plt.grid(True, 'major', 'y', ls='--', lw=.5, c='k', alpha=.3)
# plt.tight_layout()
plt.savefig("Saskatchewan-mape-cmp.png")
plt.show()