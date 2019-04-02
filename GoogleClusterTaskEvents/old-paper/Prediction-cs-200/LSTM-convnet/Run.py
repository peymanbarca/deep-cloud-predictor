from SplitData import split_data
from normalizer import normalizer
import Train_LSTM
import time
import matplotlib.pyplot as plt
import numpy as np
from time import sleep



def plot_results(predicted_data,ts_test, true_data,ts_train,y_train,rms,ms,map):
    fig = plt.figure(facecolor='white')
    ax = fig.add_subplot(311)
    ax.plot(ts_train, y_train,color='red', label='Train Data')
    ax.plot(ts_test,true_data, label='True Test Data')
    plt.plot(ts_test,predicted_data,color='green', label='Prediction , MSE  = '+str(ms)+ ' , MAPE = '+str(map))
    plt.legend()
    plt.grid()
    plt.ylim([0,1])
    #plt.xlabel('Time Symbol')
    plt.ylabel('Normalized RAM Req')
    ax = fig.add_subplot(312)
    ax.plot(ts_test, true_data, label='True Test Data')
    plt.legend()
    plt.grid()
    plt.ylim([0, 1])
    #plt.xlabel('Time Symbol')
    plt.ylabel('Normalized RAM Req')
    ax = fig.add_subplot(313)
    plt.plot(ts_test, predicted_data, color='green', label='Prediction')
    plt.legend()
    plt.grid()
    plt.ylim([0, 1])
    plt.xlabel('Time Symbol')
    plt.ylabel('Normalized RAM Req')
    plt.show()



if __name__=='__main__':
    global_start_time = time.time()
    epochs = 50
    seq_len =25
    factor=0.8
    mode=2 ## 1 for CPU, 2 for RAM

    X_train, y_train,y_train_original_part, X_test, y_test,ts_train,ts_test = Train_LSTM.load_data(seq_len,mode,factor,first_plot=True)



    model = Train_LSTM.build_model([1,25, 50, 1])
    from keras.utils.vis_utils import plot_model
    plot_model(model, to_file='model_plot.png', show_shapes=True, show_layer_names=True)

    print(np.array(X_train).shape)
    print(np.array(y_train).shape)

    print('> Data Loaded. Compiling...')
    st1=time.time()
    history=model.fit(
        X_train,
        y_train,
        batch_size=512,
        nb_epoch=epochs,
        validation_split=0.2)

    print('total training time with 50 epochs is ',time.time()-st1)

    import matplotlib.pyplot as plt

    loss = history.history['loss']
    val_loss = history.history['val_loss']
    epochs = range(1, len(loss) + 1)
    plt.figure()
    plt.plot(epochs, loss, 'bo', label='Training loss')
    plt.plot(epochs, val_loss, 'b', color='red', label='Validation loss')
    plt.title('Training and validation loss')
    plt.legend()
    plt.show()

    ''' saving the trained model'''
    model.save('model-RAM.h5')  # creates a HDF5 file 'my_model.h5'


    predicted = Train_LSTM.predict_point_by_point(model, X_test)
    print(len(predicted),len(y_test),'------------')
    del X_train,X_test,y_train
    print('-----\n--------------\n--------------------------')
    sleep(3)

    print('Training duration (s) : ', time.time() - global_start_time)

    print(' --------------------------------- ')
    from sklearn.metrics import mean_squared_error
    from math import sqrt

    rms = sqrt(mean_squared_error(y_test, predicted))
    ms = mean_squared_error(y_test, predicted)
    print('RMSE is ', rms)
    print('MSE is ', ms)
    print(' ------------------------------------')


    def mean_absolute_percentage_error(y_true, y_pred):
        ape=[]
        for k in range(len(y_true)):
            if (y_true[k] > 1e-2):
                ape.append(np.abs((y_true[k] - y_pred[k]) / (y_true[k])))

        return np.mean(np.array(ape)) * 100


    def median_absolute_percentage_error(y_true, y_pred):
        ape=[]

        for k in range(len(y_true)):
            if (y_true[k]>1e-2):
                ape.append(np.abs((y_true[k] - y_pred[k]) / (y_true[k])))

        return np.median(np.array(ape)) * 100

    map=mean_absolute_percentage_error(y_test, predicted)
    print('MAPE is ', map)
    meap = median_absolute_percentage_error(y_test, predicted)
    print('MEAPE is ', meap)

    ''' todo : denoramalize the actual and predicted data '''


    plot_results(predicted,ts_test, y_test,ts_train,y_train_original_part,rms,ms,map)





