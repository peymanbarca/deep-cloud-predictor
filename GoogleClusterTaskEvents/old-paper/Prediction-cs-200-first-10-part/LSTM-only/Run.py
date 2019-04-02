from SplitData import split_data
from normalizer import normalizer
import Train_LSTM
import time
import matplotlib.pyplot as plt
import numpy as np

#split_data(plot=True)
#normalizer(plot=True)

def plot_results(predicted_data,ts_test, true_data,ts_train,y_train,rms):
    fig = plt.figure(facecolor='white')
    ax = fig.add_subplot(311)
    ax.plot(ts_train, y_train,color='red', label='Train Data')
    ax.plot(ts_test,true_data, label='True Test Data')
    plt.plot(ts_test,predicted_data,color='green', label='Prediction-cs-200-first-10-part , RMSE  = '+str(rms))
    plt.legend()
    plt.grid()
    ax = fig.add_subplot(312)
    ax.plot(ts_test, true_data, label='True Test Data')
    plt.legend()
    plt.grid()
    ax = fig.add_subplot(313)
    plt.plot(ts_test, predicted_data, color='green', label='Prediction-cs-200-first-10-part')
    plt.legend()
    plt.grid()

    plt.show()

def plot_results_multiple(predicted_data, true_data, prediction_len):
    fig = plt.figure(facecolor='white')
    ax = fig.add_subplot(111)
    ax.plot(true_data, label='True Data')
    #Pad the list of predictions to shift it in the graph to it's correct start
    for i, data in enumerate(predicted_data):
        padding = [None for p in range(i * prediction_len)]
        plt.plot(padding + data, label='Prediction-cs-200-first-10-part')
        plt.legend()
    plt.show()

if __name__=='__main__':
    global_start_time = time.time()
    epochs = 40
    seq_len = 50
    mode=2 ## 1 for CPU, 2 for RAM

    X_train, y_train,y_train_original_part, X_test, y_test,ts_train,ts_test = Train_LSTM.load_data(seq_len,mode)



    model = Train_LSTM.build_model([1, 50, 100, 1])
    print(np.array(X_train).shape)
    print(np.array(y_train).shape)

    print('> Data Loaded. Compiling...')

    model.fit(
        X_train,
        y_train,
        batch_size=512,
        nb_epoch=epochs,
        validation_split=0.05)

    # predictions = lstm.predict_sequences_multiple(model, X_test, seq_len, 50)
    predicted = Train_LSTM.predict_point_by_point(model, X_test)
    print(len(predicted),'------------')

    print('Training duration (s) : ', time.time() - global_start_time)
    # plot_results_multiple(predictions, y_test, 50)
    print(' --------------------------------- ')
    from sklearn.metrics import mean_squared_error
    from math import sqrt

    rms = sqrt(mean_squared_error(y_test, predicted))
    print('RMSE is ', rms)
    print(' ------------------------------------')

    ''' todo : denoramalize the actual and predicted data '''


    plot_results(predicted,ts_test, y_test,ts_train,y_train_original_part,rms)





