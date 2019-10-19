from matplotlib import pyplot as plt

steps = [1,2,3,4,5]

calgary_60min_lstm=[16.14,17.81,17.45,19.74,20.41]
calgary_60min_lstm_cnn=[11.98,12.34,13.16,13.86,13.65]
calgary_60min_svr=[15.38,17.31,17.84,19.56,20.21]
calgary_60min_emd_svr=[12.35,12.76,13.02,15.41,17.86]
calgary_60min_emd_lstm=[14.70,14.67,15.98,16.22,17.14]
calgary_60min_emd_gan=[13.63,13.76,14.41,14.56,14.98]
calgary_60min_our=[12.10,12.34,12.21,12.98,13.05]

calgary_30min_lstm=[15.72,16.34,17.76,19.91,20.54]
calgary_30min_lstm_cnn=[13.41,14.74,14.96,15.81,17.02]
calgary_30min_svr=[21.31,]
calgary_30min_emd_svr=[9.17,]
calgary_30min_emd_lstm=[10.03,11.82,12.31,14.54,18.71]
calgary_30min_emd_gan=[12.29,12.71,12.67,13.54,14.14]
calgary_30min_our=[8.53,8.76,9.11,9.18,9.74]

calgary_10min_lstm=[20.28,22.36,21.75,24.31,27.75]
calgary_10min_lstm_cnn=[11.71,11.90,13.31,14.65,15.77]
calgary_10min_svr=[25.55,]
calgary_10min_emd_svr=[11.90,]
calgary_10min_emd_lstm=[14.70,15.31,15.11,16.88,18.21]
calgary_10min_emd_gan=[11.40,12.31,11.95,12.88,13.90]
calgary_10min_our=[8.76,8.79,9.81,9.95,11.01]

plt.figure(figsize=(12,14))
plt.subplot(311)
plt.plot(steps,calgary_60min_lstm,'.',color='blue',label='LSTM')
plt.plot(steps,calgary_60min_lstm_cnn,color='red',label='LSTM+CNN')
plt.plot(steps,calgary_60min_emd_lstm,'*',color='yellow',label='EMD+LSTM')
plt.plot(steps,calgary_60min_emd_gan,'-.',color='orange',label='EMD+GAN')
plt.plot(steps,calgary_60min_our,'-*',color='black',label='Ours')
plt.subplots_adjust(hspace = 0.45)
plt.title('PWS = 60 Min')
plt.ylabel('MAPE %')
#plt.xlabel('Step Ahead')
plt.grid()
plt.legend(bbox_to_anchor=(0.95, 0.8))

plt.subplot(312)
plt.plot(steps,calgary_30min_lstm,'.',color='blue',label='LSTM')
plt.plot(steps,calgary_30min_lstm_cnn,color='red',label='LSTM+CNN')
plt.plot(steps,calgary_30min_emd_lstm,'*',color='yellow',label='EMD+LSTM')
plt.plot(steps,calgary_30min_emd_gan,'-.',color='orange',label='EMD+GAN')
plt.plot(steps,calgary_30min_our,'-*',color='black',label='Ours')
plt.subplots_adjust(hspace = 0.45)
plt.title('PWS = 30 Min')
plt.ylabel('MAPE %')
#plt.xlabel('Step Ahead')
plt.grid()
plt.legend(bbox_to_anchor=(0.95, 0.8))

plt.subplot(313)
plt.plot(steps,calgary_10min_lstm,'.',color='blue',label='LSTM')
plt.plot(steps,calgary_10min_lstm_cnn,color='red',label='LSTM+CNN')
plt.plot(steps,calgary_10min_emd_lstm,'*',color='yellow',label='EMD+LSTM')
plt.plot(steps,calgary_10min_emd_gan,'-.',color='orange',label='EMD+GAN')
plt.plot(steps,calgary_10min_our,'-*',color='black',label='Ours')
plt.title('PWS = 10 Min')
plt.ylabel('MAPE %')
plt.xlabel('Forcasting Steps Ahead')
plt.grid()
plt.legend(bbox_to_anchor=(0.95, 0.8))

plt.savefig("Calgary-steps-cmp.png")
plt.show()