
import Train_LSTM
import time
import matplotlib.pyplot as plt
import numpy as np

#split_data(plot=True)
#normalizer(plot=True)

def plot_results(predicted_data,ts_test, true_data,ts_train,y_train,ms):
    fig = plt.figure(facecolor='white')
    ax = fig.add_subplot(311)
    ax.plot(ts_train, y_train,color='red', label='Train Data')
    ax.plot(ts_test,true_data, label='True Test Data')
    plt.plot(ts_test,predicted_data,color='green', label='Prediction-PWS=60min , MSE  = '+str(ms))
    plt.legend()
    plt.grid()
    ax = fig.add_subplot(312)
    ax.plot(ts_test, true_data, label='True Test Data')
    plt.legend()
    plt.grid()
    ax = fig.add_subplot(313)
    plt.plot(ts_test, predicted_data, color='green', label='Prediction')
    plt.legend()
    plt.grid()

    plt.show()


if __name__=='__main__':
    global_start_time = time.time()
    epochs = 60
    seq_len = 10

    X_train, y_train,y_train_original_part, X_test, y_test,ts_train,ts_test = Train_LSTM.load_data(seq_len)



    model = Train_LSTM.build_model([1, seq_len, 20,20,10,1])
    from keras.utils.vis_utils import plot_model
    plot_model(model, to_file='model_plot.png', show_shapes=True, show_layer_names=True)

    print(np.array(X_train).shape)
    print(np.array(y_train).shape)

    print('> Data Loaded. Compiling...')

    history =model.fit(
        X_train,
        y_train,
        batch_size=64,
        nb_epoch=epochs,
        validation_split=0.1)

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

    # predictions = lstm.predict_sequences_multiple(model, X_test, seq_len, 50)
    predicted = Train_LSTM.predict_point_by_point(model, X_test)
    print(len(predicted),'------------')

    print('Training duration (s) : ', time.time() - global_start_time)
    # plot_results_multiple(predictions, y_test, 50)
    print(' --------------------------------- ')
    from sklearn.metrics import mean_squared_error
    from math import sqrt



    ms = mean_squared_error(y_test, predicted)
    print('MSE is ', ms)
    print(' ------------------------------------')

    ''' todo : denoramalize the actual and predicted data '''

    ''' saving the trained model'''
    model.save('my_model.h5')  # creates a HDF5 file 'my_model.h5'
    del model  # deletes the existing model


    plot_results(predicted,ts_test, y_test,ts_train,y_train_original_part,ms)





