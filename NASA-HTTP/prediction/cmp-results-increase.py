from matplotlib import pyplot as plt
import numpy as np

nasa_10min_lstm=12.66
nasa_10min_lstm_cnn=13.06
nasa_10min_svr=14.15
nasa_10min_emd_svr=11.59
nasa_10min_emd_lstm=8.83
nasa_10min_emd_gan=10.13
nasa_10min_our=8.25

nasa_5min_lstm=14.37
nasa_5min_lstm_cnn=14.68
nasa_5min_svr=16.93
nasa_5min_emd_svr=9.93
nasa_5min_emd_lstm=10.26
nasa_5min_emd_gan=9.59
nasa_5min_our=9.00

nasa_1min_lstm=8.33
nasa_1min_lstm_cnn=9.94
nasa_1min_svr=9.75
nasa_1min_emd_svr=5.85
nasa_1min_emd_lstm=9.14
nasa_1min_emd_gan=6.68
nasa_1min_our=5.82

### ----------------------------
inc_lstm = (nasa_10min_our+nasa_5min_our+nasa_1min_our)/3 - (nasa_10min_lstm+nasa_5min_lstm+nasa_1min_lstm)/3
inc_lstm_cnn = (nasa_10min_our+nasa_5min_our+nasa_1min_our)/3 - (nasa_10min_lstm_cnn+nasa_5min_lstm_cnn+nasa_1min_lstm_cnn)/3
inc_svr= (nasa_10min_our+nasa_5min_our+nasa_1min_our)/3 - (nasa_10min_svr+nasa_5min_svr+nasa_1min_svr)/3
inc_emd_svr = (nasa_10min_our+nasa_5min_our+nasa_1min_our)/3 - (nasa_10min_emd_svr+nasa_5min_emd_svr+nasa_1min_emd_svr)/3
inc_emd_lstm = (nasa_10min_our+nasa_5min_our+nasa_1min_our)/3 - (nasa_10min_emd_lstm+nasa_5min_emd_lstm+nasa_1min_emd_lstm)/3
inc_emd_gan = (nasa_10min_our+nasa_5min_our+nasa_1min_our)/3 - (nasa_10min_emd_gan+nasa_5min_emd_gan+nasa_1min_emd_gan)/3

with open("inc.txt",'w') as f:
    f.write("inc_lstm : " + str(inc_lstm) +"\n")
    f.write("inc_lstm_cnn : " + str(inc_lstm_cnn)+"\n")
    f.write("inc_svr : " + str( inc_svr)+"\n")
    f.write("inc_emd_svr : " + str( inc_emd_svr)+"\n")
    f.write("inc_emd_lstm : " + str( inc_emd_lstm)+"\n")
    f.write("inc_emd_gan : " + str( inc_emd_gan)+"\n")






