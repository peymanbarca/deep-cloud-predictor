import numpy as np

def perform_knn(input_ar):
    l=len(input_ar)
    for k in range(len(input_ar)):
        if (input_ar[k]<0.1):
            if k>=3 and k<=l-4:
                input_ar[k]=(input_ar[k-3]+input_ar[k-2]+input_ar[k-1]+input_ar[k+1]+input_ar[k+2]
                             +input_ar[k+3])/6
            elif k>=l-4:
                input_ar[k] = (0.5 * input_ar[k - 1] + 0.5 * input_ar[k - 2]) / 1
            else:
                input_ar[k] = (0.5 * input_ar[k + 1] + 0.5 * input_ar[k + 2] ) / 1
            if input_ar[k] < 10e-2 and input_ar[k] >= 9e-2:
                input_ar[k] = 9e-2
            elif input_ar[k] < 9e-2 and input_ar[k] >= 8e-2:
                input_ar[k] = 8e-2
            elif input_ar[k] < 8e-2 and input_ar[k] >= 7e-2:
                input_ar[k] = 7e-2
            elif input_ar[k] < 7e-2 and input_ar[k] >= 6e-2:
                input_ar[k] = 6e-2
            elif input_ar[k] < 6e-2 and input_ar[k] >= 5e-2:
                input_ar[k] = 5e-2
            elif input_ar[k]<5e-2 and input_ar[k] >= 4e-2 :
                input_ar[k]=4e-2
            elif input_ar[k] < 4e-2 and input_ar[k] >= 3e-2 :
                input_ar[k] = 3e-2
            elif input_ar[k] < 3e-2 and input_ar[k] >= 2e-2:
                input_ar[k] = 2e-2
            elif input_ar[k] < 2e-2 and input_ar[k] >= 1e-2 :
                input_ar[k] = 1e-2
            elif input_ar[k] < 1e-2:
                input_ar[k] = 0
    return np.array(input_ar)