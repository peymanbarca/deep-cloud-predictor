from matplotlib import pyplot as plt

steps = [1,2,3,4,5]

nasa_10min_lstm=[12.66,13.67,13.89,15.02,15.90]
nasa_10min_lstm_cnn=[13.06,13.54,14.01,14.98,16.02]
nasa_10min_svr=[14.15,14.71,15.14,16.03,18.45]
nasa_10min_emd_svr=[11.59,12.21,14.33,16.22,16.80]
nasa_10min_emd_lstm=[8.83,8.94,12.11,11.88,13.22]
nasa_10min_emd_gan=[11.33,11.44,11.65,12.11,11.98]
nasa_10min_our=[8.69,8.77,8.66,8.81,9.24]

nasa_5min_lstm=[14.37,15.11,15.88,15.64,17.02]
nasa_5min_lstm_cnn=[14.68,16.12,15.88,16.88,17.67]
nasa_5min_svr=[16.93,17.24,17.88,19.44,20.21]
nasa_5min_emd_svr=[9.93,10.24,11.71,11.66,12.23]
nasa_5min_emd_lstm=[10.26,10.44,11.29,11.93,13.71]
nasa_5min_emd_gan=[9.59,9.88,9.76,10.34,10.99]
nasa_5min_our=[9.73,9.94,10.21,10.14,10.71]

nasa_1min_lstm=[8.33,9.93,10.14,10.67,12.21]
nasa_1min_lstm_cnn=[9.94,11.71,11.29,13.04,13.86]
nasa_1min_svr=[9.75,9.94,10.17,12.31,13.07]
nasa_1min_emd_svr=[5.85,7.21,9.93,9.14,11.31]
nasa_1min_emd_lstm=[9.14,9.25,10.36,11.42,13.67]
nasa_1min_emd_gan=[6.99,7.71,7.42,9.16,10.24]
nasa_1min_our=[6.85,6.36,7.42,7.13,7.96]



plt.figure(figsize=(10,14))
plt.subplot(311)
plt.plot(steps,nasa_10min_lstm,'.',color='blue',label='LSTM')
plt.plot(steps,nasa_10min_lstm_cnn,color='red',label='LSTM+CNN')
plt.plot(steps,nasa_10min_emd_lstm,'*',color='yellow',label='EMD+LSTM')
plt.plot(steps,nasa_10min_emd_gan,'-.',color='orange',label='EMD+GAN')
plt.plot(steps,nasa_10min_our,'-*',color='black',label='Ours')
plt.subplots_adjust(hspace = 0.45)
plt.title('PWS = 10 Min')
plt.ylabel('MAPE %')
# plt.xlabel('Step Ahead')
plt.grid()
plt.legend(bbox_to_anchor=(0.95, 0.8))

plt.subplot(312)
plt.plot(steps,nasa_5min_lstm,'.',color='blue',label='LSTM')
plt.plot(steps,nasa_5min_lstm_cnn,color='red',label='LSTM+CNN')
plt.plot(steps,nasa_5min_emd_lstm,'*',color='yellow',label='EMD+LSTM')
plt.plot(steps,nasa_5min_emd_gan,'-.',color='orange',label='EMD+GAN')
plt.plot(steps,nasa_5min_our,'-*',color='black',label='Ours')
plt.subplots_adjust(hspace = 0.45)
plt.title('PWS = 5 Min')
plt.ylabel('MAPE %')
# plt.xlabel('Step Ahead')
plt.grid()
plt.legend(bbox_to_anchor=(0.95, 0.8))

plt.subplot(313)
plt.plot(steps,nasa_1min_lstm,'.',color='blue',label='LSTM')
plt.plot(steps,nasa_1min_lstm_cnn,color='red',label='LSTM+CNN')
plt.plot(steps,nasa_1min_emd_lstm,'*',color='yellow',label='EMD+LSTM')
plt.plot(steps,nasa_1min_emd_gan,'-.',color='orange',label='EMD+GAN')
plt.plot(steps,nasa_1min_our,'-*',color='black',label='Ours')
plt.title('PWS = 1 Min')
plt.ylabel('MAPE %')
plt.xlabel('Forcasting Steps Ahead')
plt.grid()
plt.legend(bbox_to_anchor=(0.95, 0.8))

plt.savefig("Nasa-steps-cmp.png")
plt.show()