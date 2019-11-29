from matplotlib import pyplot as plt
import numpy as np

calgary_60min_lstm=11.82
calgary_60min_lstm_cnn=8.69
calgary_60min_svr=15.38
calgary_60min_emd_svr=12.35
calgary_60min_emd_lstm=16.96
calgary_60min_emd_gan=13.63
calgary_60min_our=11.02

calgary_30min_lstm=11.03
calgary_30min_lstm_cnn=9.25
calgary_30min_svr=21.31
calgary_30min_emd_svr=9.17
calgary_30min_emd_lstm=10.90
calgary_30min_emd_gan=12.29
calgary_30min_our=8.53

calgary_10min_lstm=10.50
calgary_10min_lstm_cnn=6.09
calgary_10min_svr=25.66
calgary_10min_emd_svr=11.90
calgary_10min_emd_lstm=11.43
calgary_10min_emd_gan=11.40
calgary_10min_our=8.76

### ----------------------------
inc_lstm = (calgary_60min_our+calgary_30min_our+calgary_10min_our)/3 - (calgary_60min_lstm+calgary_30min_lstm+calgary_10min_lstm)/3
inc_lstm_cnn = (calgary_60min_our+calgary_30min_our+calgary_10min_our)/3 - (calgary_60min_lstm_cnn+calgary_30min_lstm_cnn+calgary_10min_lstm_cnn)/3
inc_svr= (calgary_60min_our+calgary_30min_our+calgary_10min_our)/3 - (calgary_60min_svr+calgary_30min_svr+calgary_10min_svr)/3
inc_emd_svr = (calgary_60min_our+calgary_30min_our+calgary_10min_our)/3 - (calgary_60min_emd_svr+calgary_30min_emd_svr+calgary_10min_emd_svr)/3
inc_emd_lstm = (calgary_60min_our+calgary_30min_our+calgary_10min_our)/3 - (calgary_60min_emd_lstm+calgary_30min_emd_lstm+calgary_10min_emd_lstm)/3
inc_emd_gan = (calgary_60min_our+calgary_30min_our+calgary_10min_our)/3 - (calgary_60min_emd_gan+calgary_30min_emd_gan+calgary_10min_emd_gan)/3

with open("inc.txt",'w') as f:
    f.write("inc_lstm : " + str(inc_lstm) +"\n")
    f.write("inc_lstm_cnn : " + str(inc_lstm_cnn)+"\n")
    f.write("inc_svr : " + str( inc_svr)+"\n")
    f.write("inc_emd_svr : " + str( inc_emd_svr)+"\n")
    f.write("inc_emd_lstm : " + str( inc_emd_lstm)+"\n")
    f.write("inc_emd_gan : " + str( inc_emd_gan)+"\n")






