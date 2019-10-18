
import Train_LSTM
import time
import matplotlib.pyplot as plt
import numpy as np
from time import sleep



def plot_results(predicted_data,ts_test, true_data,ts_train,y_train,rms,ms,map,meap,rmsre):
    fig = plt.figure(facecolor='white', figsize=(10, 8))
    ax = fig.add_subplot(311)
    ax.plot(ts_train, y_train,color='red', label='Train Data')
    ax.plot(ts_test,true_data, label='True Test Data',alpha=0.4)
    plt.plot(ts_test,predicted_data,color='green',alpha=0.8, label='Prediction , MSE  = '+str(ms)+ ' , MAPE = '+str(map) +\
               ' \n MEAPE = '+str(meap)+ ' , RMSRE = '+str(rmsre) + ', RMSE='+str(rms))
    plt.legend()
    plt.grid()
    plt.ylim([0,1])
    plt.ylabel('Normalized Req')
    ax = fig.add_subplot(312)
    ax.plot(ts_test, true_data, label='True Test Data')
    plt.legend()
    plt.grid()
    plt.ylim([0, 1])
    plt.ylabel('Normalized Req')
    ax = fig.add_subplot(313)
    plt.plot(ts_test, predicted_data, color='green', label='Prediction')
    plt.legend()
    plt.grid()
    plt.ylim([0, 1])
    plt.xlabel('Time Symbol')
    plt.ylabel('Normalized Req')
    plt.savefig(
            'results'
            '/Normalized' + '.png',
            dpi=700)

    plt.pause(3)
    plt.close()


def plot_results_denormalize(predicted_data,ts_test, true_data,ts_train,y_train,rms,ms,map,meap,rmsre):
    fig = plt.figure(facecolor='white', figsize=(10, 8))
    ax = fig.add_subplot(311)
    ax.plot(ts_train, y_train,color='red', label='Train Data')
    ax.plot(ts_test,true_data, label='True Test Data',alpha=0.4)
    plt.plot(ts_test,predicted_data,color='green',alpha=0.8, label='Prediction , MSE  = '+str(ms)+ ' , MAPE = '+str(map) +\
               ' \n MEAPE = '+str(meap)+ ' , RMSRE = '+str(rmsre) + ', RMSE='+str(rms))
    plt.legend()
    plt.grid()
    #plt.ylim([0,1])
    plt.ylabel(' Original Req')
    ax = fig.add_subplot(312)
    ax.plot(ts_test, true_data, label='True Test Data')
    plt.legend()
    plt.grid()
    plt.ylabel(' Original Req')
    ax = fig.add_subplot(313)
    plt.plot(ts_test, predicted_data, color='green', label='Prediction')
    plt.legend()
    plt.grid()
    #plt.ylim([0, 1])
    plt.xlabel('Time Symbol')
    plt.ylabel(' Original Req')
    plt.savefig(
            'results'
            '/Main' + '.png',
            dpi=700)

    plt.pause(3)
    plt.close()



if __name__=='__main__':
    global_start_time = time.time()
    epochs = 50
    seq_len =25


    X_train, y_train,y_train_original_part, X_test, y_test,ts_train,ts_test,minMaxScaler = \
        Train_LSTM.load_data(seq_len)



    model = Train_LSTM.build_model([1,seq_len, 50,10, 3])
    #from keras.utils.vis_utils import plot_model
    #plot_model(model, to_file='model_plot.png', show_shapes=True, show_layer_names=True)

    print(np.array(X_train).shape)
    print(np.array(y_train).shape)

    print('> Data Loaded. Compiling...')
    st1=time.time()
    history=model.fit(
        X_train,
        y_train,
        batch_size=256,
        nb_epoch=epochs,
        validation_split=0.1)

    print('total training time  is ',time.time()-st1)

    import matplotlib.pyplot as plt

    loss = history.history['loss']
    val_loss = history.history['val_loss']
    epochs = range(1, len(loss) + 1)
    plt.figure()
    plt.plot(epochs, loss, 'bo', label='Training loss')
    plt.plot(epochs, val_loss, 'b', color='red', label='Validation loss')
    plt.title('Training and validation loss')
    plt.legend()
    plt.savefig(
            'results'
            '/Normalized_Train_Loss' + '.png',
            dpi=700)
    plt.pause(3)
    plt.close()


    ''' saving the trained model'''
    model.save('model.h5')  # creates a HDF5 file



    predicted = Train_LSTM.predict_point_by_point(model, X_test)
    print(len(predicted), len(y_test), '------------',y_test.shape,predicted.shape)

    # ---------- resize from matrix to vector for evaluations --------
    predicted = np.reshape(predicted, (predicted.size,))
    y_test = np.reshape(y_test, (y_test.size,))
    print(len(predicted),len(y_test),'------------')

    del X_train,X_test
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

        ape = []
        for k in range(len(y_true)):
            # if abs(y_true[k])!=0 and k not in z1 and k not in z2:
            if abs(y_pred[k]) > 1e-3 and abs(y_true[k]) > 1e-3:
                ape.append(abs((y_true[k] - y_pred[k]) / y_true[k]))
        plt.hist(ape, bins='auto',color='orange')
        plt.xlabel('MAPE')
        plt.ylabel('frequency')
        plt.grid()
        plt.savefig(
                'results'
                '/MAPE' + '.png', dpi=600)


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

    map=mean_absolute_percentage_error(y_test, predicted)
    print('MAPE is ', map)
    meap = median_absolute_percentage_error(y_test, predicted)
    print('MEAPE is ', meap)
    rmsre = mean_percentage_r_error(y_test, predicted)
    print('MPRE is ', rmsre)

    #plot_results(predicted, ts_test, y_test, ts_train, y_train_original_part, rms, ms, map,meap,rmsre)

    ''' todo : denoramalize the actual and predicted data '''
    print('Denormalizeing Data ....')

    from denorm import denorm_v2

    y_test_revert = denorm_v2(y_test,minMaxScaler)
    print('-----------------------', len(y_test), len(y_test_revert))
    y_train_revert = denorm_v2(y_train,minMaxScaler)
    print('-----------------------', len(y_train), len(y_train_revert))
    y_train_original_revert = denorm_v2(y_train_original_part,minMaxScaler)
    print('-----------------------', len(y_train_original_part), len(y_train_original_revert))
    y_pred_revert = denorm_v2(predicted,minMaxScaler)
    print('-----------------------', len(predicted), len(y_pred_revert))

    y_test_revert = np.reshape(y_test_revert, (y_test_revert.size,))
    y_pred_revert = np.reshape(y_pred_revert, (y_pred_revert.size,))

    rms = sqrt(mean_squared_error(y_test_revert, y_pred_revert))
    ms = mean_squared_error(y_test_revert, y_pred_revert)
    print('RMSE is ', rms)
    print('MSE is ', ms)
    print(' ------------------------------------')

    map=mean_absolute_percentage_error(y_test_revert, y_pred_revert)
    print('MAPE is ', map)
    meap = median_absolute_percentage_error(y_test_revert, y_pred_revert)
    print('MEAPE is ', meap)
    rmsre = mean_percentage_r_error(y_test_revert, y_pred_revert)
    print('RMSRE is ', rmsre)

    # plot_results_denormalize(y_pred_revert, ts_test,y_test_revert, ts_train, y_train_original_revert,
    #                          rms, ms, map, meap, rmsre)








