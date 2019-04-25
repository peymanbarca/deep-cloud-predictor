
import Train_LSTM
import time
import matplotlib.pyplot as plt
import numpy as np
import psycopg2
import numpy as np
from denorm import denorm_v2

hostname = 'localhost'
username = 'postgres'
password = 'inter2010'
database = 'load_cloud'

conn = psycopg2.connect(host=hostname, user=username, password=password, dbname=database)
cur=conn.cursor()

def plot_results(imf,predicted_data,ts_test, true_data,ts_train,y_train,ms,map,
                 y_train_revert,y_test_revert,y_predicted_revert,map_denormalize,rms_denormalize):


    # ------------------- plot normalize Data --------------
    fig = plt.figure(facecolor='white',figsize=(10, 8))
    ax = fig.add_subplot(211)
    ax.plot(ts_train, y_train,color='red', label='Train Data')
    ax.plot(ts_test,true_data, label='True Test Data')
    plt.plot(ts_test,predicted_data,'-.',color='green',
             label='Prediction-PWS=5sec IMF= ' + str(imf) +' , MSE  = '+str(ms) + ' MAPE = ' +str(map))
    plt.legend()
    plt.grid()
    ax = fig.add_subplot(212)
    ax.plot(ts_test, true_data, label='True Test Data')
    plt.plot(ts_test, predicted_data,'-.', color='green', label='Prediction')
    plt.legend()
    plt.grid()
    plt.savefig('results/imf'+str(imf)+'/IMF_'+str(imf)+'.png', dpi=900)
    plt.pause(3)
    plt.close()


    # ------------------- plot denormalize Data --------------
    fig = plt.figure(facecolor='white',figsize=(10, 8))
    ax = fig.add_subplot(211)
    ax.plot(ts_train, y_train_revert, color='red', label='Train Data')
    ax.plot(ts_test, y_test_revert, label='True Test Data')
    plt.plot(ts_test, y_predicted_revert, color='green',
             label='Prediction-PWS=5sec , RMSE  = ' + str(rms_denormalize))
    plt.legend()
    plt.grid()
    ax = fig.add_subplot(212)
    ax.plot(ts_test, y_test_revert, label='True Test Data')
    plt.plot(ts_test, y_predicted_revert, color='green', label='Prediction')
    plt.legend()
    plt.grid()
    plt.savefig('results/imf' + str(imf) + '/IMF_Original_' + str(imf) + '.png', dpi=900)
    plt.pause(3)
    plt.close()

def write_prediction_to_db(ts_test,y_test,y_pred,imf):
    cur.execute('delete from nasa_http_emd_10min  where imf_index=%s and  num_req_pred is not null', \
                ([int(imf)]))

    conn.commit()
    cur.execute('update nasa_http_emd_10min set num_req_pred=null where imf_index=%s', \
                ( [int(imf)]))
    conn.commit()
    for k in range(len(ts_test)):
        cur.execute('insert into nasa_http_emd_10min (ts,num_of_req,imf_index,num_req_pred) values(%s,%s,%s,%s) ',
                    (int(ts_test[k]),int(y_test[k]),imf,float(y_pred[k])))
    conn.commit()

if __name__=='__main__':
    global_start_time = time.time()
    epochs = 100
    seq_len = 10
    imf_index=15
    norm_version=1 # v2= MinMaxScaler(0,1) , v1=MaxAbsScaler(-1,1)

    X_train, y_train,y_train_original_part, X_test, y_test,ts_train,ts_test,MaxAbsScalerObj =\
        Train_LSTM.load_data(seq_len,imf_index,norm_version)



    model = Train_LSTM.build_model([1, seq_len, 20,1])
    from keras.utils.vis_utils import plot_model
    #plot_model(model, to_file='model_plot.png', show_shapes=True, show_layer_names=True)

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
    plt.pause(3)
    plt.close()


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


    def mean_absolute_percentage_error(y_true, y_pred):
        #y_true, y_pred = np.abs(y_true)+max(y_true),np.abs(y_pred)+max(y_pred)
        ape = []
        for k in range(len(y_true)):
            if abs(y_pred[k]) > 1e-2 and abs(y_true[k]) > 1e-2:
                ape.append(    abs((y_pred[k] - y_true[k])  / y_true[k] )  )

        return np.mean(np.array(ape)) * 100


    def median_absolute_percentage_error(y_true, y_pred):
        #y_true, y_pred = np.abs(y_true) + max(y_true), np.abs(y_pred) + max(y_pred)
        ape = []

        for k in range(len(y_true)):
            if abs(y_pred[k]) > 1e-2 and abs(y_true[k]) > 1e-2:
                ape.append(  abs((y_pred[k] - y_true[k]) / y_true[k])  )

        return np.median(np.array(ape)) * 100


    map = mean_absolute_percentage_error(y_test, predicted)
    print('MAPE is ', map)
    meap = median_absolute_percentage_error(y_test, predicted)
    print('MEAPE is ', meap)

    ''' denoramalize the actual and predicted data '''
    min_train, max_train, ts_train_revert = denorm_v2(y_train_original_part,MaxAbsScalerObj)
    min_test, max_test, ts_test_revert = denorm_v2(y_test, MaxAbsScalerObj)
    min_predicted, max_predicted, ts_predicted_revert = denorm_v2(predicted, MaxAbsScalerObj)
    print('min_train=%s , max_train=%s',(min_train,max_train))
    print('min_test=%s , max_test=%s', (min_test, max_test))
    print('min_predicted=%s , max_predicted=%s', (min_predicted, max_predicted))

    map_denormalize = mean_absolute_percentage_error(ts_test_revert, ts_predicted_revert)
    #print('MAPE in original scale is ', map_denormalize)
    meap_denormalize = median_absolute_percentage_error(ts_test_revert, ts_predicted_revert)
    #print('MEAPE in original scale is ', meap_denormalize)
    rms_denormalize = sqrt(mean_squared_error(ts_test_revert, ts_predicted_revert))
    #print('RMSE in original scale is ', rms_denormalize)
    print(' ------------------------------------')

    ''' saving the trained model'''
    model.save('my_model.h5')  # creates a HDF5 file 'my_model.h5'
    del model  # deletes the existing model

    plt.plot(ts_test, ts_predicted_revert, color='green', label='Prediction')
    plt.plot(ts_test, ts_test_revert, color='cyan', label='Test')
    plt.pause(3)
    plt.close()


    print('writing to DB ! ...')
    write_prediction_to_db(ts_test,ts_test_revert,ts_predicted_revert,imf_index)

    plot_results(imf_index,predicted,ts_test, y_test,ts_train,y_train_original_part,ms,map,
                 ts_train_revert,ts_test_revert,ts_predicted_revert,map_denormalize,rms_denormalize)





