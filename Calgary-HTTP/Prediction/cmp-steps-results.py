from matplotlib import pyplot as plt

steps = [1,2,3,4,5]
xi = list(range(1,len(steps)+1))

calgary_60min_lstm=[11.82,13.81,14.45,15.74,17.41]
calgary_60min_lstm_cnn=[8.69,9.34,11.16,13.86,15.65]
calgary_60min_svr=[15.38,]
calgary_60min_emd_svr=[12.35,]
calgary_60min_emd_lstm=[16.96,17.67,17.98,18.72,20.14]
calgary_60min_emd_gan=[13.63,13.76,14.41,14.56,14.98]
calgary_60min_our=[11.02,11.34,12.21,12.98,13.05]

calgary_30min_lstm=[11.03,12.34,14.76,15.91,17.54]
calgary_30min_lstm_cnn=[9.25,11.74,12.96,13.81,15.02]
calgary_30min_svr=[21.31,]
calgary_30min_emd_svr=[9.17,]
calgary_30min_emd_lstm=[10.03,11.82,12.31,14.54,18.71]
calgary_30min_emd_gan=[12.29,12.71,12.67,13.54,14.14]
calgary_30min_our=[8.53,8.76,9.11,9.18,9.74]

calgary_10min_lstm=[10.50,11.36,13.75,14.31,16.75]
calgary_10min_lstm_cnn=[6.09,7.90,8.31,10.65,11.77]
calgary_10min_svr=[25.55,]
calgary_10min_emd_svr=[11.90,]
calgary_10min_emd_lstm=[11.43,12.31,14.11,15.88,16.21]
calgary_10min_emd_gan=[11.40,12.31,11.95,12.88,13.90]
calgary_10min_our=[8.76,8.79,9.81,9.95,11.01]

plt.figure(figsize=(8,12))
plt.rc('legend', fontsize=7)

plt.subplot(311)
plt.plot(steps,calgary_60min_lstm,'.-',color='blue',label='LSTM')
plt.plot(steps,calgary_60min_lstm_cnn,'--',color='purple',label='LSTM+CNN')
plt.plot(steps,calgary_60min_emd_lstm,color='red',label='EMD+LSTM')
plt.plot(steps,calgary_60min_emd_gan,'-.',color='brown',label='EMD+GAN')
plt.plot(steps,calgary_60min_our,'-*',color='black',label='ELG')
plt.subplots_adjust(hspace = 0.45)
plt.ylabel('MAPE %')
plt.xlabel('A) PWS = 60 Minutes')
plt.grid()
plt.xticks(xi, steps)
plt.legend(bbox_to_anchor=(0.97, 0.8))

plt.subplot(312)
plt.plot(steps,calgary_30min_lstm,'.-',color='blue',label='LSTM')
plt.plot(steps,calgary_30min_lstm_cnn,'--',color='purple',label='LSTM+CNN')
plt.plot(steps,calgary_30min_emd_lstm,color='red',label='EMD+LSTM')
plt.plot(steps,calgary_30min_emd_gan,'-.',color='brown',label='EMD+GAN')
plt.plot(steps,calgary_30min_our,'-*',color='black',label='ELG')
plt.subplots_adjust(hspace = 0.45)
plt.ylabel('MAPE %')
plt.xlabel('B) PWS = 30 Minutes')
plt.grid()
plt.xticks(xi, steps)
plt.legend(bbox_to_anchor=(0.97, 0.8))

plt.subplot(313)
plt.plot(steps,calgary_10min_lstm,'.-',color='blue',label='LSTM')
plt.plot(steps,calgary_10min_lstm_cnn,'--',color='purple',label='LSTM+CNN')
plt.plot(steps,calgary_10min_emd_lstm,color='red',label='EMD+LSTM')
plt.plot(steps,calgary_10min_emd_gan,'-.',color='brown',label='EMD+GAN')
plt.plot(steps,calgary_10min_our,'-*',color='black',label='ELG')
plt.ylabel('MAPE %')
plt.xlabel('C) PWS = 10 Minutes\n\nForcasting Steps Ahead')
plt.grid()
plt.xticks(xi, steps)
plt.legend(bbox_to_anchor=(0.97, 0.8))

plt.savefig("Calgary-steps-cmp.png",dpi=700)
plt.show()