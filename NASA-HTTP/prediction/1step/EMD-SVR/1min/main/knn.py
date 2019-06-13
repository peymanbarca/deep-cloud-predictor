import numpy as np
import math

def perform_knn(input_ar):
    l=len(input_ar)
    for k in range(len(input_ar)):
        if (abs(input_ar[k])<0.1):
            # if k>=3 and k<=l-4:
            #     input_ar[k]=(input_ar[k-3]+input_ar[k-2]+input_ar[k-1]+input_ar[k+1]+input_ar[k+2]
            #                  +input_ar[k+3])/6
            # elif k>=l-4:
            #     input_ar[k] = (0.5 * input_ar[k - 1] + 0.5 * input_ar[k - 2]) / 1
            # else:
            #     input_ar[k] = (0.5 * input_ar[k + 1] + 0.5 * input_ar[k + 2] ) / 1
            if abs(input_ar[k]) < 10e-2 and abs(input_ar[k]) >= 9e-2:
                input_ar[k] = 9e-2* math.copysign(1, input_ar[k])
            elif abs(input_ar[k]) < 9e-2 and abs(input_ar[k]) >= 8e-2:
                input_ar[k] = 8e-2* math.copysign(1, input_ar[k])
            elif abs(input_ar[k]) < 8e-2 and abs(input_ar[k]) >= 7e-2:
                input_ar[k] = 7e-2* math.copysign(1, input_ar[k])
            elif abs(input_ar[k]) < 7e-2 and abs(input_ar[k]) >= 6e-2:
                input_ar[k] = 6e-2* math.copysign(1, input_ar[k])
            elif abs(input_ar[k]) < 6e-2 and abs(input_ar[k]) >= 5e-2:
                input_ar[k] = 5e-2* math.copysign(1, input_ar[k])
            elif abs(input_ar[k])<5e-2 and abs(input_ar[k]) >= 4e-2 :
                input_ar[k]=4e-2* math.copysign(1, input_ar[k])
            elif abs(input_ar[k]) < 4e-2 and abs(input_ar[k]) >= 3e-2 :
                input_ar[k] = 3e-2* math.copysign(1, input_ar[k])
            elif abs(input_ar[k]) < 3e-2 and abs(input_ar[k]) >= 2e-2:
                input_ar[k] = 2e-2* math.copysign(1, input_ar[k])
            elif abs(input_ar[k]) < 2e-2 and abs(input_ar[k]) >= 1e-2 :
                input_ar[k] = 1e-2* math.copysign(1, input_ar[k])
            elif abs(input_ar[k]) < 1e-2:
                input_ar[k] = 0
        if (abs(input_ar[k]) < 0.2 and abs(input_ar[k]) >= 0.1):

            if abs(input_ar[k]) < 20e-2 and abs(input_ar[k]) >= 19e-2:
                input_ar[k] = 19e-2 * math.copysign(1, input_ar[k])
            elif abs(input_ar[k]) < 19e-2 and abs(input_ar[k]) >= 18e-2:
                input_ar[k] = 18e-2 * math.copysign(1, input_ar[k])
            elif abs(input_ar[k]) < 18e-2 and abs(input_ar[k]) >= 17e-2:
                input_ar[k] = 17e-2 * math.copysign(1, input_ar[k])
            elif abs(input_ar[k]) < 17e-2 and abs(input_ar[k]) >= 16e-2:
                input_ar[k] = 16e-2 * math.copysign(1, input_ar[k])
            elif abs(input_ar[k]) < 16e-2 and abs(input_ar[k]) >= 15e-2:
                input_ar[k] = 15e-2 * math.copysign(1, input_ar[k])
            elif abs(input_ar[k]) < 15e-2 and abs(input_ar[k]) >= 14e-2:
                input_ar[k] = 14e-2 * math.copysign(1, input_ar[k])
            elif abs(input_ar[k]) < 14e-2 and abs(input_ar[k]) >= 13e-2:
                input_ar[k] = 13e-2 * math.copysign(1, input_ar[k])
            elif abs(input_ar[k]) < 13e-2 and abs(input_ar[k]) >= 12e-2:
                input_ar[k] = 12e-2 * math.copysign(1, input_ar[k])
            elif abs(input_ar[k]) < 12e-2 and abs(input_ar[k]) >= 11e-2:
                input_ar[k] = 11e-2 * math.copysign(1, input_ar[k])
            elif abs(input_ar[k]) < 11e-2:
                input_ar[k] = 0.1
        if (abs(input_ar[k]) < 0.3 and abs(input_ar[k]) >= 0.2):

            if abs(input_ar[k]) < 30e-2 and abs(input_ar[k]) >= 29e-2:
                input_ar[k] = 29e-2 * math.copysign(1, input_ar[k])
            elif abs(input_ar[k]) < 29e-2 and abs(input_ar[k]) >= 28e-2:
                input_ar[k] = 28e-2 * math.copysign(1, input_ar[k])
            elif abs(input_ar[k]) < 28e-2 and abs(input_ar[k]) >= 27e-2:
                input_ar[k] = 27e-2 * math.copysign(1, input_ar[k])
            elif abs(input_ar[k]) < 27e-2 and abs(input_ar[k]) >= 26e-2:
                input_ar[k] = 26e-2 * math.copysign(1, input_ar[k])
            elif abs(input_ar[k]) < 26e-2 and abs(input_ar[k]) >= 25e-2:
                input_ar[k] = 25e-2 * math.copysign(1, input_ar[k])
            elif abs(input_ar[k]) < 25e-2 and abs(input_ar[k]) >= 24e-2:
                input_ar[k] = 24e-2 * math.copysign(1, input_ar[k])
            elif abs(input_ar[k]) < 24e-2 and abs(input_ar[k]) >= 23e-2:
                input_ar[k] = 23e-2 * math.copysign(1, input_ar[k])
            elif abs(input_ar[k]) < 23e-2 and abs(input_ar[k]) >= 22e-2:
                input_ar[k] = 22e-2 * math.copysign(1, input_ar[k])
            elif abs(input_ar[k]) < 22e-2 and abs(input_ar[k]) >= 21e-2:
                input_ar[k] = 21e-2 * math.copysign(1, input_ar[k])
            elif abs(input_ar[k]) < 21e-2:
                input_ar[k] = 0.2
        # if (abs(input_ar[k]) < 0.4 and abs(input_ar[k]) >= 0.3):
        #
        #     if abs(input_ar[k]) < 40e-2 and abs(input_ar[k]) >= 39e-2:
        #         input_ar[k] = 39e-2 * math.copysign(1, input_ar[k])
        #     elif abs(input_ar[k]) < 39e-2 and abs(input_ar[k]) >= 38e-2:
        #         input_ar[k] = 38e-2 * math.copysign(1, input_ar[k])
        #     elif abs(input_ar[k]) < 38e-2 and abs(input_ar[k]) >= 37e-2:
        #         input_ar[k] = 37e-2 * math.copysign(1, input_ar[k])
        #     elif abs(input_ar[k]) < 37e-2 and abs(input_ar[k]) >= 36e-2:
        #         input_ar[k] = 36e-2 * math.copysign(1, input_ar[k])
        #     elif abs(input_ar[k]) < 36e-2 and abs(input_ar[k]) >= 35e-2:
        #         input_ar[k] = 35e-2 * math.copysign(1, input_ar[k])
        #     elif abs(input_ar[k]) < 35e-2 and abs(input_ar[k]) >= 34e-2:
        #         input_ar[k] = 34e-2 * math.copysign(1, input_ar[k])
        #     elif abs(input_ar[k]) < 34e-2 and abs(input_ar[k]) >= 33e-2:
        #         input_ar[k] = 33e-2 * math.copysign(1, input_ar[k])
        #     elif abs(input_ar[k]) < 33e-2 and abs(input_ar[k]) >= 32e-2:
        #         input_ar[k] = 32e-2 * math.copysign(1, input_ar[k])
        #     elif abs(input_ar[k]) < 32e-2 and abs(input_ar[k]) >= 31e-2:
        #         input_ar[k] = 31e-2 * math.copysign(1, input_ar[k])
        #     elif abs(input_ar[k]) < 31e-2:
        #         input_ar[k] = 0.3
        # if (abs(input_ar[k]) < 0.5 and abs(input_ar[k]) >= 0.4):
        #
        #     if abs(input_ar[k]) < 50e-2 and abs(input_ar[k]) >= 49e-2:
        #         input_ar[k] = 49e-2 * math.copysign(1, input_ar[k])
        #     elif abs(input_ar[k]) < 49e-2 and abs(input_ar[k]) >= 48e-2:
        #         input_ar[k] = 48e-2 * math.copysign(1, input_ar[k])
        #     elif abs(input_ar[k]) < 48e-2 and abs(input_ar[k]) >= 47e-2:
        #         input_ar[k] = 47e-2 * math.copysign(1, input_ar[k])
        #     elif abs(input_ar[k]) < 47e-2 and abs(input_ar[k]) >= 46e-2:
        #         input_ar[k] = 46e-2 * math.copysign(1, input_ar[k])
        #     elif abs(input_ar[k]) < 46e-2 and abs(input_ar[k]) >= 45e-2:
        #         input_ar[k] = 45e-2 * math.copysign(1, input_ar[k])
        #     elif abs(input_ar[k]) < 45e-2 and abs(input_ar[k]) >= 44e-2:
        #         input_ar[k] = 44e-2 * math.copysign(1, input_ar[k])
        #     elif abs(input_ar[k]) < 44e-2 and abs(input_ar[k]) >= 43e-2:
        #         input_ar[k] = 43e-2 * math.copysign(1, input_ar[k])
        #     elif abs(input_ar[k]) < 43e-2 and abs(input_ar[k]) >= 42e-2:
        #         input_ar[k] = 42e-2 * math.copysign(1, input_ar[k])
        #     elif abs(input_ar[k]) < 42e-2 and abs(input_ar[k]) >= 41e-2:
        #         input_ar[k] = 41e-2 * math.copysign(1, input_ar[k])
        #     elif abs(input_ar[k]) < 41e-2:
        #         input_ar[k] = 0.4
        # if (abs(input_ar[k]) < 0.6 and abs(input_ar[k]) >= 0.5):
        #
        #     if abs(input_ar[k]) < 60e-2 and abs(input_ar[k]) >= 59e-2:
        #         input_ar[k] = 59e-2 * math.copysign(1, input_ar[k])
        #     elif abs(input_ar[k]) < 59e-2 and abs(input_ar[k]) >= 58e-2:
        #         input_ar[k] = 58e-2 * math.copysign(1, input_ar[k])
        #     elif abs(input_ar[k]) < 58e-2 and abs(input_ar[k]) >= 57e-2:
        #         input_ar[k] = 57e-2 * math.copysign(1, input_ar[k])
        #     elif abs(input_ar[k]) < 57e-2 and abs(input_ar[k]) >= 56e-2:
        #         input_ar[k] = 56e-2 * math.copysign(1, input_ar[k])
        #     elif abs(input_ar[k]) < 56e-2 and abs(input_ar[k]) >= 55e-2:
        #         input_ar[k] = 55e-2 * math.copysign(1, input_ar[k])
        #     elif abs(input_ar[k]) < 55e-2 and abs(input_ar[k]) >= 54e-2:
        #         input_ar[k] = 54e-2 * math.copysign(1, input_ar[k])
        #     elif abs(input_ar[k]) < 54e-2 and abs(input_ar[k]) >= 53e-2:
        #         input_ar[k] = 53e-2 * math.copysign(1, input_ar[k])
        #     elif abs(input_ar[k]) < 53e-2 and abs(input_ar[k]) >= 52e-2:
        #         input_ar[k] = 52e-2 * math.copysign(1, input_ar[k])
        #     elif abs(input_ar[k]) < 52e-2 and abs(input_ar[k]) >= 51e-2:
        #         input_ar[k] = 51e-2 * math.copysign(1, input_ar[k])
        #     elif abs(input_ar[k]) < 51e-2:
        #         input_ar[k] = 0.5
        # if (abs(input_ar[k]) < 0.7 and abs(input_ar[k]) >= 0.6):
        #
        #     if abs(input_ar[k]) < 70e-2 and abs(input_ar[k]) >= 69e-2:
        #         input_ar[k] = 69e-2 * math.copysign(1, input_ar[k])
        #     elif abs(input_ar[k]) < 69e-2 and abs(input_ar[k]) >= 68e-2:
        #         input_ar[k] = 68e-2 * math.copysign(1, input_ar[k])
        #     elif abs(input_ar[k]) < 68e-2 and abs(input_ar[k]) >= 67e-2:
        #         input_ar[k] = 67e-2 * math.copysign(1, input_ar[k])
        #     elif abs(input_ar[k]) < 67e-2 and abs(input_ar[k]) >= 66e-2:
        #         input_ar[k] = 66e-2 * math.copysign(1, input_ar[k])
        #     elif abs(input_ar[k]) < 66e-2 and abs(input_ar[k]) >= 65e-2:
        #         input_ar[k] = 65e-2 * math.copysign(1, input_ar[k])
        #     elif abs(input_ar[k]) < 65e-2 and abs(input_ar[k]) >= 64e-2:
        #         input_ar[k] = 64e-2 * math.copysign(1, input_ar[k])
        #     elif abs(input_ar[k]) < 64e-2 and abs(input_ar[k]) >= 63e-2:
        #         input_ar[k] = 63e-2 * math.copysign(1, input_ar[k])
        #     elif abs(input_ar[k]) < 63e-2 and abs(input_ar[k]) >= 62e-2:
        #         input_ar[k] = 62e-2 * math.copysign(1, input_ar[k])
        #     elif abs(input_ar[k]) < 62e-2 and abs(input_ar[k]) >= 61e-2:
        #         input_ar[k] = 61e-2 * math.copysign(1, input_ar[k])
        #     elif abs(input_ar[k]) < 61e-2:
        #         input_ar[k] = 0.6
        # if (abs(input_ar[k]) < 0.8 and abs(input_ar[k]) >= 0.7):
        #
        #     if abs(input_ar[k]) < 80e-2 and abs(input_ar[k]) >= 79e-2:
        #         input_ar[k] = 79e-2 * math.copysign(1, input_ar[k])
        #     elif abs(input_ar[k]) < 79e-2 and abs(input_ar[k]) >= 78e-2:
        #         input_ar[k] = 78e-2 * math.copysign(1, input_ar[k])
        #     elif abs(input_ar[k]) < 78e-2 and abs(input_ar[k]) >= 77e-2:
        #         input_ar[k] = 77e-2 * math.copysign(1, input_ar[k])
        #     elif abs(input_ar[k]) < 77e-2 and abs(input_ar[k]) >= 76e-2:
        #         input_ar[k] = 76e-2 * math.copysign(1, input_ar[k])
        #     elif abs(input_ar[k]) < 76e-2 and abs(input_ar[k]) >= 75e-2:
        #         input_ar[k] = 75e-2 * math.copysign(1, input_ar[k])
        #     elif abs(input_ar[k]) < 75e-2 and abs(input_ar[k]) >= 74e-2:
        #         input_ar[k] = 74e-2 * math.copysign(1, input_ar[k])
        #     elif abs(input_ar[k]) < 74e-2 and abs(input_ar[k]) >= 73e-2:
        #         input_ar[k] = 73e-2 * math.copysign(1, input_ar[k])
        #     elif abs(input_ar[k]) < 73e-2 and abs(input_ar[k]) >= 72e-2:
        #         input_ar[k] = 72e-2 * math.copysign(1, input_ar[k])
        #     elif abs(input_ar[k]) < 72e-2 and abs(input_ar[k]) >= 71e-2:
        #         input_ar[k] = 71e-2 * math.copysign(1, input_ar[k])
        #     elif abs(input_ar[k]) < 71e-2:
        #         input_ar[k] = 0.7
        # if (abs(input_ar[k]) < 0.9 and abs(input_ar[k]) >= 0.8):
        #
        #     if abs(input_ar[k]) < 90e-2 and abs(input_ar[k]) >= 89e-2:
        #         input_ar[k] = 89e-2 * math.copysign(1, input_ar[k])
        #     elif abs(input_ar[k]) < 79e-2 and abs(input_ar[k]) >= 88e-2:
        #         input_ar[k] = 88e-2 * math.copysign(1, input_ar[k])
        #     elif abs(input_ar[k]) < 78e-2 and abs(input_ar[k]) >= 87e-2:
        #         input_ar[k] = 87e-2 * math.copysign(1, input_ar[k])
        #     elif abs(input_ar[k]) < 77e-2 and abs(input_ar[k]) >= 86e-2:
        #         input_ar[k] = 86e-2 * math.copysign(1, input_ar[k])
        #     elif abs(input_ar[k]) < 76e-2 and abs(input_ar[k]) >= 85e-2:
        #         input_ar[k] = 85e-2 * math.copysign(1, input_ar[k])
        #     elif abs(input_ar[k]) < 75e-2 and abs(input_ar[k]) >= 84e-2:
        #         input_ar[k] = 84e-2 * math.copysign(1, input_ar[k])
        #     elif abs(input_ar[k]) < 74e-2 and abs(input_ar[k]) >= 83e-2:
        #         input_ar[k] = 83e-2 * math.copysign(1, input_ar[k])
        #     elif abs(input_ar[k]) < 73e-2 and abs(input_ar[k]) >= 82e-2:
        #         input_ar[k] = 82e-2 * math.copysign(1, input_ar[k])
        #     elif abs(input_ar[k]) < 72e-2 and abs(input_ar[k]) >= 81e-2:
        #         input_ar[k] = 81e-2 * math.copysign(1, input_ar[k])
        #     elif abs(input_ar[k]) < 81e-2:
        #         input_ar[k] = 0.8
    return np.array(input_ar)