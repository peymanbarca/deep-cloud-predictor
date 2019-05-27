
def norm_v3(ts):
    min1=min(ts)
    max1=max(ts)
    ts_normalize = ts / (max1 - min1)
    min2 = min(ts_normalize)
    max2 = max(ts_normalize)
    return min1,max1,min2,max2,ts_normalize

def norm_v1(ts):
    min1=min(ts)
    max1=max(ts)
    from sklearn import preprocessing
    MaxAbsScalerObj = preprocessing.MaxAbsScaler()
    ts_normalize = MaxAbsScalerObj.fit_transform(ts.reshape(-1, 1))
    print('-----------------------', len(ts), len(ts_normalize))
    min2 = min(ts_normalize)
    max2 = max(ts_normalize)
    return min1, max1, min2, max2, ts_normalize, MaxAbsScalerObj

def norm_v2(ts):
    min1=min(ts)
    max1=max(ts)
    from sklearn import preprocessing
    MaxAbsScalerObj = preprocessing.MinMaxScaler()
    ts_normalize = MaxAbsScalerObj.fit_transform(ts.reshape(-1, 1))
    print('-----------------------',len(ts),len(ts_normalize))
    min2 = min(ts_normalize)
    max2 = max(ts_normalize)
    return min1,max1,min2,max2,ts_normalize,MaxAbsScalerObj

def norm_v2_single(ts):
    from sklearn import preprocessing
    MaxAbsScalerObj = preprocessing.MinMaxScaler()
    ts_normalize = MaxAbsScalerObj.fit_transform(ts.reshape(-1, 1))
    return ts_normalize