from matplotlib import pyplot as plt
import numpy as np

calgary_60min_lstm=[11.82]
calgary_60min_lstm_cnn=[8.69]
calgary_60min_svr=[15.38]
calgary_60min_emd_svr=[12.35]
calgary_60min_emd_lstm=[16.96]
calgary_60min_emd_gan=[13.63]
calgary_60min_our=[11.02]

calgary_30min_lstm=[11.03]
calgary_30min_lstm_cnn=[9.25]
calgary_30min_svr=[21.31]
calgary_30min_emd_svr=[9.17]
calgary_30min_emd_lstm=[10.90]
calgary_30min_emd_gan=[12.29]
calgary_30min_our=[8.53]

calgary_10min_lstm=[10.50]
calgary_10min_lstm_cnn=[6.09]
calgary_10min_svr=[25.66]
calgary_10min_emd_svr=[11.90]
calgary_10min_emd_lstm=[11.43]
calgary_10min_emd_gan=[11.40]
calgary_10min_our=[8.76]

# --------------------#
# data to plot
n_groups = 3
Ours = (calgary_60min_our[0],calgary_30min_our[0],calgary_10min_our[0])
EMD_LSTM = (calgary_60min_emd_lstm[0],calgary_30min_emd_lstm[0],calgary_10min_emd_lstm[0])
LSTM = (calgary_60min_lstm[0], calgary_30min_lstm[0], calgary_10min_lstm[0])
EMD_GAN = (calgary_60min_emd_gan[0],calgary_30min_emd_gan[0],calgary_10min_emd_gan[0])
EMD_SVR = (calgary_60min_emd_svr[0],calgary_30min_emd_svr[0],calgary_10min_emd_svr[0])
CNN_LSTM = (calgary_60min_lstm_cnn[0],calgary_30min_lstm_cnn[0],calgary_10min_lstm_cnn[0])
SVR = (calgary_60min_svr[0],calgary_30min_svr[0],calgary_10min_svr[0])

plt.rc('legend', fontsize=6)

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
label='EMD+SVR')

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
plt.ylim([5,26])
plt.title('Calgary HTTP Trace')
plt.xticks(index + bar_width, ('60', '30', '10'))
plt.legend()
plt.grid(True, 'major', 'y', ls='--', lw=.5, c='k', alpha=.3)
# plt.tight_layout()
plt.savefig("Calgary-mape-cmp.png",dpi=700)
plt.show()