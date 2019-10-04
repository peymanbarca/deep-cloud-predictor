
def denorm_v1(min1,max1,min2,max2,ts_normalize):
    ts=ts_normalize * (max1-min1)
    return min1,max1,min2,max2,ts

def denorm_v2(ts_normalize,MaxAbsScalerObj):
    from sklearn import preprocessing

    ts_revert = MaxAbsScalerObj.inverse_transform(ts_normalize.reshape(-1, 1))
    min2 = min(ts_revert)
    max2 = max(ts_revert)
    print(min2,max2)
    return  ts_revert

