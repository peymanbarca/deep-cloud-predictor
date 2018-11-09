import numpy as np

def perform_knn(input_ar,param):
    l=len(input_ar)
    for k in range(len(input_ar)):
        if (input_ar[k]<1e-2):
            if k>param and k<l-param:
                input_ar[k]=np.mean(input_ar[k-param:k+param])
            else:
                input_ar[k] = (0.5 * input_ar[k + 1] + 0.5 * input_ar[k + 2] ) / 1
            if(input_ar[k]<1e-2):
                input_ar[k]=0
    return np.array(input_ar)