from matplotlib import pyplot as plt

steps = [1,2,3,4,5]
xi = list(range(1,len(steps)+1))

sasc_30min_lstm=[14.69,14.98,15.50,16.12,17.74]
sasc_30min_lstm_cnn=[7.79,7.56,9.14,10.33,11.07]
sasc_30min_svr=[11.39,12.35,12.21,13.96,15.85]
sasc_30min_emd_svr=[15.78,14.31,15.98,17.84,17.62]
sasc_30min_emd_lstm=[9.56,10.14,10.21,11.76,12.32]
sasc_30min_emd_gan=[12.78,12.98,13.09,14.19,16.06]
sasc_30min_our=[7.81,8.34,8.55,8.46,9.11]

sasc_20min_lstm=[13.68,13.99,14.76,15.66,17.04]
sasc_20min_lstm_cnn=[8.55,9.94,11.31,12.22,15.98]
sasc_20min_svr=[11.18,12.31,12.88,13.77,15.01]
sasc_20min_emd_svr=[14.30,14.22,14.76,15.41,15.88]
sasc_20min_emd_lstm=[9.69,9.90,10.34,11.71,13.54]
sasc_20min_emd_gan=[12.20,12.71,12.34,12.56,12.90]
sasc_20min_our=[8.63,9.14,9.16,9.71,9.44]

sasc_10min_lstm=[11.19,11.79,12.41,13.65,15.01]
sasc_10min_lstm_cnn=[8.36,9.41,10.94,11.24,13.41]
sasc_10min_svr=[16.71,15.98,17.65,18.02,20.71]
sasc_10min_emd_svr=[11.20,11,44,12,91,13.04,13.88]
sasc_10min_emd_lstm=[6.62,7.23,8.38,9.96,11.04]
sasc_10min_emd_gan=[9.40,9.56,9.67,9.89,10.02]
sasc_10min_our=[5.94,6.34,6.90,7.41,7.89]

plt.figure(figsize=(8,12))
plt.rc('legend', fontsize=7)

plt.subplot(311)
plt.plot(steps,sasc_30min_lstm,'.-',color='blue',label='LSTM')
plt.plot(steps,sasc_30min_lstm_cnn,'--',color='purple',label='LSTM+CNN')
plt.plot(steps,sasc_30min_emd_lstm,color='red',label='EMD+LSTM')
plt.plot(steps,sasc_30min_emd_gan,'-.',color='brown',label='EMD+GAN')
plt.plot(steps,sasc_30min_our,'-*',color='black',label='ELG')
plt.subplots_adjust(hspace = 0.45)
plt.ylabel('MAPE %')
plt.xlabel('A) PWS = 30 Minutes')
plt.grid()
plt.xticks(xi, steps)
plt.legend(bbox_to_anchor=(0.97, 0.8))

plt.subplot(312)
plt.plot(steps,sasc_20min_lstm,'.-',color='blue',label='LSTM')
plt.plot(steps,sasc_20min_lstm_cnn,'--',color='purple',label='LSTM+CNN')
plt.plot(steps,sasc_20min_emd_lstm,color='red',label='EMD+LSTM')
plt.plot(steps,sasc_20min_emd_gan,'-.',color='brown',label='EMD+GAN')
plt.plot(steps,sasc_20min_our,'-*',color='black',label='ELG')
plt.subplots_adjust(hspace = 0.45)
plt.ylabel('MAPE %')
plt.xlabel('B) PWS = 20 Minutes')
plt.grid()
plt.xticks(xi, steps)
plt.legend(bbox_to_anchor=(0.97, 0.8))

plt.subplot(313)
plt.plot(steps,sasc_10min_lstm,'.-',color='blue',label='LSTM')
plt.plot(steps,sasc_10min_lstm_cnn,'--',color='purple',label='LSTM+CNN')
plt.plot(steps,sasc_10min_emd_lstm,color='red',label='EMD+LSTM')
plt.plot(steps,sasc_10min_emd_gan,'-.',color='brown',label='EMD+GAN')
plt.plot(steps,sasc_10min_our,'-*',color='black',label='ELG')
plt.ylabel('MAPE %')
plt.xlabel('C) PWS = 10 Minutes\n\n Forcasting Steps Ahead')
plt.grid()
plt.xticks(xi, steps)
plt.legend(bbox_to_anchor=(0.97, 0.8))

plt.savefig("sasc-steps-cmp.png",dpi=700)
plt.show()