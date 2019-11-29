from matplotlib import pyplot as plt
import numpy as np

sasc_30min_lstm=14.69
sasc_30min_lstm_cnn=7.79
sasc_30min_svr=11.39
sasc_30min_emd_svr=15.78
sasc_30min_emd_lstm=9.56
sasc_30min_emd_gan=12.78
sasc_30min_our=7.81

sasc_20min_lstm=13.68
sasc_20min_lstm_cnn=8.55
sasc_20min_svr=11.18
sasc_20min_emd_svr=14.30
sasc_20min_emd_lstm=9.69
sasc_20min_emd_gan=12.20
sasc_20min_our=8.63

sasc_10min_lstm=11.19
sasc_10min_lstm_cnn=8.36
sasc_10min_svr=16.71
sasc_10min_emd_svr=11.20
sasc_10min_emd_lstm=6.62
sasc_10min_emd_gan=9.40
sasc_10min_our=5.94

### ----------------------------
inc_lstm = (sasc_30min_our+sasc_20min_our+sasc_10min_our)/3 - (sasc_30min_lstm+sasc_20min_lstm+sasc_10min_lstm)/3
inc_lstm_cnn = (sasc_30min_our+sasc_20min_our+sasc_10min_our)/3 - (sasc_30min_lstm_cnn+sasc_20min_lstm_cnn+sasc_10min_lstm_cnn)/3
inc_svr= (sasc_30min_our+sasc_20min_our+sasc_10min_our)/3 - (sasc_30min_svr+sasc_20min_svr+sasc_10min_svr)/3
inc_emd_svr = (sasc_30min_our+sasc_20min_our+sasc_10min_our)/3 - (sasc_30min_emd_svr+sasc_20min_emd_svr+sasc_10min_emd_svr)/3
inc_emd_lstm = (sasc_30min_our+sasc_20min_our+sasc_10min_our)/3 - (sasc_30min_emd_lstm+sasc_20min_emd_lstm+sasc_10min_emd_lstm)/3
inc_emd_gan = (sasc_30min_our+sasc_20min_our+sasc_10min_our)/3 - (sasc_30min_emd_gan+sasc_20min_emd_gan+sasc_10min_emd_gan)/3

with open("inc.txt",'w') as f:
    f.write("inc_lstm : " + str(inc_lstm) +"\n")
    f.write("inc_lstm_cnn : " + str(inc_lstm_cnn)+"\n")
    f.write("inc_svr : " + str( inc_svr)+"\n")
    f.write("inc_emd_svr : " + str( inc_emd_svr)+"\n")
    f.write("inc_emd_lstm : " + str( inc_emd_lstm)+"\n")
    f.write("inc_emd_gan : " + str( inc_emd_gan)+"\n")






