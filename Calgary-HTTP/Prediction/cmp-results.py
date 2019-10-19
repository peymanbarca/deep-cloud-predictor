from matplotlib import pyplot as plt
import numpy as np

calgary_60min_lstm=[16.14]
calgary_60min_lstm_cnn=[11.98]
calgary_60min_svr=[15.38]
calgary_60min_emd_svr=[12.35]
calgary_60min_emd_lstm=[14.70]
calgary_60min_emd_gan=[13.63]
calgary_60min_our=[12.10]

calgary_30min_lstm=[15.72]
calgary_30min_lstm_cnn=[13.41]
calgary_30min_svr=[21.31]
calgary_30min_emd_svr=[9.17]
calgary_30min_emd_lstm=[10.03]
calgary_30min_emd_gan=[12.29]
calgary_30min_our=[8.53]

calgary_10min_lstm=[20.28]
calgary_10min_lstm_cnn=[11.71]
calgary_10min_svr=[25.55]
calgary_10min_emd_svr=[11.90]
calgary_10min_emd_lstm=[14.70]
calgary_10min_emd_gan=[11.40]
calgary_10min_our=[8.76]

# --------------------#
# data to plot
n_groups = 3
Ours = (12.10,8.53,8.76)
EMD_LSTM = (14.70,10.03,14.70)
LSTM = (16.14, 15.72, 20.28)
EMD_GAN = (13.63,12.29,11.40)
EMD_SVR = (12.35,9.17,11.90)
CNN_LSTM = (11.98,13.41,11.71)
SVR = (15.38,21.31,25.55)

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
plt.title('Calgary HTTP Trace')
plt.xticks(index + bar_width, ('60 min', '30 min', '10 min'))
plt.legend()

# plt.tight_layout()
plt.savefig("Calgary-mape-cmp.png")
plt.show()