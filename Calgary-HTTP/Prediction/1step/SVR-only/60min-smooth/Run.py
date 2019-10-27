from normalizer import normalizer
import numpy as np
from sklearn.svm import SVC,SVR
import pandas as pd
from matplotlib import pyplot as plt
from denorm import denorm_v2
import psycopg2


hostname = 'localhost'
username = 'postgres'
password = 'inter2010'
database = 'load_cloud'

conn = psycopg2.connect(host=hostname, user=username, password=password, dbname=database)
cur=conn.cursor()


norm_Ver=1 # 1 = maxAbsSclaer (-1,1) 2=minMaxSclaer (0,1)
seq_lag=20
train_factor=0.9




ts,num_req_normalize,MaxAbsScalerObj=normalizer(norm_Ver,False)
reqs = [j for i in num_req_normalize for j in i]
#print(len(reqs),len(ts))

df = pd.DataFrame({'requests':np.array(reqs)})
print(df.shape)

# define function for create N lags
def create_lags(df, N):
    for i in range(N):
        df['Lag' + str(i+1)] = df.requests.shift(i+1)
    return df

# create  lags
df = create_lags(df,seq_lag)


df = df.dropna()
#print(df.head(10))

# create X and y
y = df.requests.values
# X = df.ts.values
X = df.iloc[:, 1:].values

#print('some samples of sequenses are:\n')
# print(reqs[:60])
# print(X[0])
# print(X[1])
# print(X[2])
# print(y[0])
# print(y[1])
# print(y[2])
# print('********************')

print('length of sequenses are :')
print('X : ',len(X),'*',len(X[0]),' Y : ',len(y),'*','1')

# Train on %s of the data
train_idx = int(len(df) * train_factor)

# create train and test data
X_train, y_train, X_test, y_test = X[:train_idx], y[:train_idx], X[train_idx:], y[train_idx:]
ts_train,ts_test=ts[:train_idx], ts[train_idx:][seq_lag:]

# fit and predict
clf = SVR(kernel='rbf',C=1,epsilon=0.1)
clf.fit(X_train, y_train)

y_pred=clf.predict(X_test)
score=clf.score(X_test,y_pred)
print(score)
print('length of output data are : ')
print(len(X_test),len(y_pred),len(y_test),len(ts_test),len(ts_train),len(X_train),len(y_train))
print('-----')


from sklearn.metrics import mean_squared_error
from math import sqrt
ms = mean_squared_error(y_test, y_pred)
rms=sqrt(ms)
print('MSE is ', ms)
print(' ------------------------------------')


def mean_absolute_percentage_error(y_true, y_pred):
    ape = []
    for k in range(len(y_true)):
        # if abs(y_true[k])!=0 and k not in z1 and k not in z2:
        if abs(y_pred[k]) > 1e-3 and abs(y_true[k]) > 1e-3:
            ape.append(abs((y_true[k] - y_pred[k]) / y_true[k]))
    plt.hist(ape, bins='auto', color='orange')
    plt.xlabel('MAPE')
    plt.ylabel('frequency')
    plt.grid()

    plt.pause(3)
    plt.close()
    ape = sorted(ape)
    indexes = np.where(ape < np.percentile(ape, 90))[0]
    ape = [ape[k] for k in indexes]
    # print(ape)

    return np.mean(np.array(ape)) * 100


def median_absolute_percentage_error(y_true, y_pred):
    ape = []
    for k in range(len(y_true)):
        if abs(y_pred[k]) > 1e-3 and abs(y_true[k]) > 1e-3:
            # if abs(y_true[k])!=0  and k not in z1 and k not in z2:
            ape.append(abs((y_pred[k] - y_true[k]) / y_true[k]))
    ape = sorted(ape)
    indexes = np.where(ape < np.percentile(ape, 90))[0]
    ape = [ape[k] for k in indexes]
    return np.median(np.array(ape)) * 100


def mean_percentage_r_error(y_true, y_pred):
    ape = []
    for k in range(len(y_true)):
        if abs(y_pred[k]) > 1e-3 and abs(y_true[k]) > 1e-3:
            # if abs(y_true[k])!=0  and k not in z1 and k not in z2:
            ape.append(pow(((y_true[k] - y_pred[k]) / y_true[k]), 2))
    ape = sorted(ape)
    indexes = np.where(ape < np.percentile(ape, 90))[0]
    ape = [ape[k] for k in indexes]
    return sqrt(np.mean(np.array(ape)))

try:
    map = mean_absolute_percentage_error(y_test, y_pred)
    print('MAPE is ', map)
    meap = median_absolute_percentage_error(y_test, y_pred)
    print('MEAPE is ', meap)
    rmsre = mean_percentage_r_error(y_test, y_pred)
    print('RMSRE is ', rmsre)
except:
    map=0
    meap=0
    rmsre=0

# fig = plt.figure(facecolor='white',figsize=(10, 8))
# plt.subplot(3,1,1)
# plt.plot(ts_train,y_train,label='Train Data',alpha=0.8,color='red')
# plt.plot(ts_test,y_pred,label='Prediction Data',alpha=0.9)
# plt.plot(ts_test,y_test,label='Test Data',alpha=0.3)
# plt.legend()
# plt.subplot(3,1,2)
# plt.plot(y_test,label='Test Data',color='orange')
# plt.legend()
# plt.subplot(3,1,3)
# plt.plot(y_pred,label='Prediction Real Data, MAPE = %.4f%% ,  RMSE=%.4f  ,\n  MEAPE=%.4f%% , RMSRE=%.4f '% (map,rms,meap,rmsre))
# plt.legend()

fig = plt.figure(facecolor='white',figsize=(12, 7))
ax = fig.add_subplot(211)
plt.plot(ts_train,y_train,color='red',label='Training Set')
plt.plot(ts_test, y_test, color='blue',alpha=0.4,
         label='Testing Set')
plt.plot(ts_test,y_pred,'-.',color='green',
         label='Prediction',alpha=0.9)
plt.ylabel('Number of Requests')
plt.legend()
plt.grid()
ax = fig.add_subplot(212)
plt.subplots_adjust(hspace = 0.3)
plt.plot(ts_test,y_test,'-',color='blue',label='Testing Set',alpha=0.4)
plt.plot(ts_test,y_pred,'-',color='green',alpha=0.9,
         label='Prediction')
plt.plot(ts_test,y_test-y_pred,'-',color='black',alpha=0.7,
         label=('Error'))
plt.title(' MAPE = %.4f%% ,  RMSE=%.4f ,  MEAPE=%.4f%%, RMSRE=%4f '% (map,rms,meap,rmsre)
          ,backgroundcolor='black',color='white')
plt.xlabel('Time Index')
plt.ylabel('Number of Requests')
plt.legend()
plt.grid()
plt.savefig('results'
          '/normalize' + '.png', dpi=900)
plt.pause(5)
plt.close()


''' denoramalize the actual and predicted data '''
min_train, max_train, ts_train_revert = denorm_v2(y_train, MaxAbsScalerObj)
min_test, max_test, ts_test_revert = denorm_v2(y_test, MaxAbsScalerObj)
min_predicted, max_predicted, ts_predicted_revert = denorm_v2(y_pred, MaxAbsScalerObj)
print('min_test=%s , max_test=%s', (min_test, max_test))
print('min_predicted=%s , max_predicted=%s', (min_predicted, max_predicted))

ts_train_revert=np.reshape(ts_train_revert, (ts_train_revert.size,))
ts_test_revert=np.reshape(ts_test_revert, (ts_test_revert.size,))
ts_predicted_revert=np.reshape(ts_predicted_revert, (ts_predicted_revert.size,))

try:
    map_denormalize = mean_absolute_percentage_error(ts_test_revert, ts_predicted_revert)
    print('MAPE in original scale is ', map_denormalize)
    meap_denormalize = median_absolute_percentage_error(ts_test_revert, ts_predicted_revert)
    print('MEAPE in original scale is ', meap_denormalize)
    rms_denormalize = sqrt(mean_squared_error(ts_test_revert, ts_predicted_revert))
    print('RMSE in original scale is ', rms_denormalize)
    rmsre_denorm = mean_percentage_r_error(ts_test_revert, ts_predicted_revert)
    print(' ------------------------------------')
except:
    map_denormalize=0
    meap_denormalize=0
    rmsre_denorm=0
    rms_denormalize = sqrt(mean_squared_error(ts_test_revert, ts_predicted_revert))

# fig = plt.figure(facecolor='white',figsize=(10, 8))
# plt.subplot(3,1,1)
# plt.plot(ts_predicted_revert,label='Prediction Real Data',alpha=0.8)
# plt.plot(ts_test_revert,label='Test Real Data',alpha=0.3)
# plt.legend()
# plt.subplot(3,1,2)
# plt.plot(ts_test_revert,label='Test Real Data',color='orange')
# plt.legend()
# plt.subplot(3,1,3)
# plt.plot(ts_predicted_revert,label='Prediction Real Data, MAPE = %.4f%% ,\n '
#                                    ' RMSE=%.4f  , RMSRE=%.4f  '% (map_denormalize,rms_denormalize,rmsre_denorm))
# plt.legend()

fig = plt.figure(facecolor='white',figsize=(12, 7))
ax = fig.add_subplot(211)
plt.plot(ts_train,ts_train_revert,color='red',label='Training Set')
plt.plot(ts_test, ts_test_revert, color='blue',alpha=0.4,
         label='Testing Set')
plt.plot(ts_test,ts_predicted_revert,'-.',color='green',
         label='Prediction',alpha=0.9)
plt.ylabel('Number of Requests')
plt.legend()
plt.grid()
ax = fig.add_subplot(212)
plt.subplots_adjust(hspace = 0.3)
plt.plot(ts_test,ts_test_revert,'-',color='blue',label='Testing Set',alpha=0.4)
plt.plot(ts_test,ts_predicted_revert,'-',color='green',alpha=0.9,
         label='Prediction')
plt.plot(ts_test,ts_test_revert-ts_predicted_revert,'-',color='black',alpha=0.7,
         label=('Error'))
plt.title(' MAPE = %.4f%% ,  RMSE=%.4f ,  MEAPE=%.4f%%, RMSRE=%4f '% (map,rms,meap,rmsre)
          ,backgroundcolor='black',color='white')
plt.xlabel('Time Index')
plt.ylabel('Number of Requests')
plt.legend()
plt.grid()
plt.savefig('results'
             + '/original' + '.png', dpi=900)
plt.pause(5)
plt.close()




