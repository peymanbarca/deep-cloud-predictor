from normalizer import normalizer
import numpy as np
from sklearn.svm import SVC,SVR
import pandas as pd
from matplotlib import pyplot as plt
from denorm import denorm_v2
from norm import norm_v2_single
import psycopg2


hostname = 'localhost'
username = 'postgres'
password = 'inter2010'
database = 'cloud_load'

conn = psycopg2.connect(host=hostname, user=username, password=password, dbname=database)
cur=conn.cursor()


norm_Ver=1
imf_index=20
ts,num_req_normalize,MaxAbsScalerObj=normalizer(imf_index,norm_Ver,False)
reqs = [j for i in num_req_normalize for j in i]
print(len(reqs),len(ts))

df = pd.DataFrame({'requests':np.array(reqs)})
print(df.shape)

# define function for create N lags
def create_lags(df, N):
    for i in range(N):
        df['Lag' + str(i+1)] = df.requests.shift(i+1)
    return df

# create 10 lags
df = create_lags(df,20)

# the first 10 days will have missing values. can't use them.
df = df.dropna()
print(df.head(10))

# create X and y
y = df.requests.values
# X = df.ts.values
X = df.iloc[:, 1:].values


# Train on 90% of the data
train_idx = int(len(df) * .9)

# create train and test data
X_train, y_train, X_test, y_test = X[:train_idx], y[:train_idx], X[train_idx:], y[train_idx:]
ts_train,ts_test=ts[:train_idx], ts[train_idx:][20:]

# fit and predict
clf = SVR()
clf.fit(X_train, y_train)

y_pred=clf.predict(X_test)
print(len(X_test),len(y_pred),len(y_test),len(ts_test),len(ts_train),len(X_train),len(y_train))
print('-----')


from sklearn.metrics import mean_squared_error
from math import sqrt
ms = mean_squared_error(y_test, y_pred)
rms=sqrt(ms)
print('MSE is ', ms)
print(' ------------------------------------')


def mean_absolute_percentage_error(y_true, y_pred):
    y_true, y_pred = norm_v2_single(y_true),norm_v2_single(y_pred)
    y_true, y_pred = np.array(y_true),  np.array(y_pred)
    ape = []
    for k in range(len(y_true)):
        if y_pred[k] > 1e-3 and y_true[k] > 1e-3:
            ape.append(    abs((y_pred[k] - y_true[k])  / y_true[k] ))

    return np.mean(np.array(ape)) * 100

def mean_percentage_error(y_true, y_pred):
    y_true, y_pred = norm_v2_single(y_true),norm_v2_single(y_pred)
    y_true, y_pred = np.array(y_true),  np.array(y_pred)
    ape = []
    for k in range(len(y_true)):
        if y_pred[k] > 1e-3 and y_true[k] > 1e-3:
            ape.append(    ((y_pred[k] - y_true[k])  / y_true[k] ))

    return np.mean(np.array(ape)) * 100


def median_absolute_percentage_error(y_true, y_pred):
    y_true, y_pred = norm_v2_single(y_true), norm_v2_single(y_pred)
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    ape = []
    for k in range(len(y_true)):
        if y_pred[k] > 1e-3 and y_true[k] > 1e-3:
            ape.append(abs((y_pred[k] - y_true[k]) / y_true[k]))

    return np.median(np.array(ape)) * 100


mpe = mean_percentage_error(y_test, y_pred)
map = mean_absolute_percentage_error(y_test, y_pred)
print('MAPE is ', map)
meap = median_absolute_percentage_error(y_test, y_pred)
print('MEAPE is ', meap)

fig = plt.figure(facecolor='white',figsize=(10, 8))
plt.subplot(3,1,1)
plt.plot(ts_train,y_train,label='Train Data',alpha=0.8,color='red')
plt.plot(ts_test,y_pred,label='Prediction Data',alpha=0.9)
plt.plot(ts_test,y_test,label='Test Data',alpha=0.3)
plt.legend()
plt.subplot(3,1,2)
plt.plot(y_test,label='Test Data',color='orange')
plt.legend()
plt.subplot(3,1,3)
plt.plot(y_pred,label='Prediction Real Data, MAPE = %.4f ,  RMSE=%.4f , MPE=%.4f ,  MEAPE=%.4f '% (map,rms,mpe,meap))
plt.legend()
plt.savefig('results/imf' + str(imf_index) + '/normalize' + '.png', dpi=900)
plt.pause(5)
plt.close()


''' denoramalize the actual and predicted data '''
min_test, max_test, ts_test_revert = denorm_v2(y_test, MaxAbsScalerObj)
min_predicted, max_predicted, ts_predicted_revert = denorm_v2(y_pred, MaxAbsScalerObj)
print('min_test=%s , max_test=%s', (min_test, max_test))
print('min_predicted=%s , max_predicted=%s', (min_predicted, max_predicted))

map_denormalize = mean_absolute_percentage_error(ts_test_revert, ts_predicted_revert)
print('MAPE in original scale is ', map_denormalize)
meap_denormalize = median_absolute_percentage_error(ts_test_revert, ts_predicted_revert)
print('MEAPE in original scale is ', meap_denormalize)
rms_denormalize = sqrt(mean_squared_error(ts_test_revert, ts_predicted_revert))
print('RMSE in original scale is ', rms_denormalize)
mpe_denorm = mean_percentage_error(ts_test_revert, ts_predicted_revert)
print(' ------------------------------------')

fig = plt.figure(facecolor='white',figsize=(10, 8))
plt.subplot(3,1,1)
plt.plot(ts_predicted_revert,label='Prediction Real Data',alpha=0.8)
plt.plot(ts_test_revert,label='Test Real Data',alpha=0.3)
plt.legend()
plt.subplot(3,1,2)
plt.plot(ts_test_revert,label='Test Real Data',color='orange')
plt.legend()
plt.subplot(3,1,3)
plt.plot(ts_predicted_revert,label='Prediction Real Data, MAPE = %.4f , '
                                   ' RMSE=%.4f , MPE=%.4f ,  MEAPE=%.4f '% (map_denormalize,rms_denormalize,mpe_denorm,meap_denormalize))
plt.legend()
plt.savefig('results/imf' + str(imf_index) + '/original' + '.png', dpi=900)
plt.pause(5)
plt.close()

# def write_prediction_to_db(ts_test,y_test,y_pred,imf):
#     cur.execute('delete from nasa_http_emd_5min  where imf_index=%s and  num_req_pred is not null', \
#                 ([int(imf)]))
#
#     conn.commit()
#     cur.execute('update nasa_http_emd_5min set num_req_pred=null where imf_index=%s', \
#                 ( [int(imf)]))
#     conn.commit()
#     for k in range(len(ts_test)):
#         cur.execute('insert into nasa_http_emd_5min (ts,num_of_req,imf_index,num_req_pred) values(%s,%s,%s,%s) ',
#                     (int(ts_test[k]),int(y_test[k]),imf,float(y_pred[k])))
#     conn.commit()
#
# write_prediction_to_db(ts_test,y_test,y_pred,imf_index)



