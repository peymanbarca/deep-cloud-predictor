from __future__ import print_function, division


from keras.layers import Input, Dense, Reshape, Flatten, Dropout,Concatenate
from keras.layers import BatchNormalization, Activation, ZeroPadding2D
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import UpSampling2D, Conv2D,Conv1D,MaxPooling1D
from keras.models import Sequential, Model
from keras.optimizers import Adam
from keras.backend import reshape,expand_dims
from keras.layers.recurrent import LSTM,Recurrent,SimpleRNN
from keras.utils import plot_model
from keras.backend import tensorflow_backend as K

import matplotlib.pyplot as plt

import sys
import tensorflow as tf

import numpy as np
import Train_LSTM
from denorm import denorm_v2
import time
import psycopg2


hostname = 'localhost'
username = 'postgres'
password = 'inter2010'
database = 'load_cloud'

conn = psycopg2.connect(host=hostname, user=username, password=password, dbname=database)
cur=conn.cursor()






class GAN():
    def __init__(self):


        self.T = 1
        self.seq_len=30
        self.g_in_shape=(self.seq_len,1,)
        self.d_in_shape = (1+self.seq_len,1, ) # output of G and input to D

        optimizer = Adam(lr=0.0002)



        # Build and compile the discriminator
        self.discriminator = self.build_discriminator()
        self.discriminator.compile(loss='binary_crossentropy',
            optimizer=optimizer,
            metrics=['accuracy'])

        # Build the generator
        self.generator = self.build_generator()
        self.generator.compile(loss='mse',
                                   optimizer=optimizer)

        # The generator takes x_train & y_train as input and generates two vectors
        # iterate one point by point based on a length ( T )
        g_in = Input(shape=self.g_in_shape)
        y = Input(shape=(seq_len, 1,))
        g_out = self.generator(inputs=[g_in,y])
        d_in = g_out
        print(d_in.shape)
        print('-----------------------------------------***')




        # For the combined model we will only train the generator
        self.discriminator.trainable = False

        # The discriminator takes generated images as input and determines validity
        validity = self.discriminator(d_in)
        self.valid_vector_shape=validity.shape[1]


        # The combined model  (stacked generator and discriminator)
        # Trains the generator to fool the discriminator
        self.combined = Model(inputs=[g_in,y],outputs= validity)
        self.combined.compile(loss='binary_crossentropy', optimizer=optimizer)


    def build_generator(self):
        # the process is iterating one point by point ...
        # input is X_train & Y_train
        # output is y_hat_vector & y_bar_vector

        g_in = Input(shape=self.g_in_shape)



        tmp=LSTM(
            input_shape=(self.seq_len, 1),
            output_dim=self.seq_len,
            return_sequences=True, dropout=0.1, recurrent_dropout=0.5)(g_in)
        # model.add(Dropout(0.2))


        tmp=LSTM(
            20,
            return_sequences=False, dropout=0.1, recurrent_dropout=0.5)(tmp)
        # model.add(Dropout(0.2))

        # model.add(Dense(
        #     output_dim=layers[4],activation='relu'))

        tmp=Dense(
            output_dim=1)(tmp)
        tmp=Activation("linear")(tmp)

        tmp=Reshape(target_shape=(1, 1,))(tmp)

        y = Input(shape=(seq_len, 1,))
        print(y.shape, tmp.shape)
        g_out=Concatenate(axis=1)([y, tmp])
        print(g_out.shape)
        print('------------')

        model = Model(inputs=[g_in,y],outputs=g_out)

        model.summary()
        plot_model(model, to_file='g.png',show_layer_names=True,show_shapes=True)

        return model

    def build_discriminator(self):

        d_in = Input(shape=self.d_in_shape)


        #tmp=Flatten(input_shape=self.d_in_shape)(d_in)
        tmp=Conv1D(32, 4, activation='relu',
                                input_shape=(None, 1))(d_in)
        tmp=MaxPooling1D(3)(tmp)
        tmp=Conv1D(64, 4, activation='relu')(tmp)
        # tmp=MaxPooling1D(3)(tmp)
        # tmp=Conv1D(128, 4, activation='relu')(tmp)
        tmp=Dense(128)(tmp)
        tmp=LeakyReLU(alpha=0.2)(tmp)
        validity=Dense(2, activation='sigmoid')(tmp)
        model = Model(d_in, validity)
        model.summary()
        plot_model(model, to_file='d.png',show_layer_names=True,show_shapes=True)


        return model



    def train(self, epochs,batchsize=128,verbose=False):

        # with tf.Session(config=tf.ConfigProto(intra_op_parallelism_threads=4, inter_op_parallelism_threads=4)) as sess:
        #     K.set_session(sess)
            # The generator takes x_train & y_train as input and generates two vectors
            # iterate one point by point based on a length ( T )
            print('*******************************************************************')
            print('         ***************** Running *********************')

            d_loss_reals=[]
            d_loss_fakes=[]
            d_losses=[]
            g_losses=[]
            g_loss_predictions=[]

            for epoch in range(epochs):
                t1=time.time()
                print('<<<---------------------------------------------------------------------------','epoch ',
                      epoch,'------------------------------------>>>')
                l=len(X_train)//batchsize
                print('having ',l,' batches!')
                for i in range(l):

                    d_loss_reals_tmp = []
                    d_loss_fakes_tmp = []
                    d_losses_tmp = []
                    g_losses_tmp = []
                    g_loss_predictions_tmp = []
                    # ---------------------
                    #  Train Discriminator
                    # ---------------------


                    X = X_train[i*batchsize:(i+1)*batchsize]
                    Y = X
                    y_true = y_train[i*batchsize:(i+1)*batchsize]  # y_T+1
                    y_true=np.expand_dims(y_true,3)
                    g_in = X
                    print('shape of X is ',X.shape,'shape of Y is ',Y.shape,
                          'shape of y_true is ',y_true.shape) if i==0 else None #if verbose==True else None
                    g_out = self.generator.predict([g_in,Y])  # which is y_hat_T+1



                    y_hat_vector = g_out
                    y_vector = np.concatenate((Y, y_true),axis=1)


                    print('\n size of output of Generator:\n') if i==0 else None
                    print('y_hat: ',y_hat_vector.shape,'=',g_out.shape,'y_bar: ',y_vector.shape) \
                        if i == 0  else None

                    #d_in = y_hat_vector  # shape = (1,T+1,1)

                    # Adversarial ground truths
                    valid = np.ones(shape=(batchsize, self.valid_vector_shape,2))
                    fake = np.zeros(shape=(batchsize, self.valid_vector_shape,2))




                    # Train the discriminator
                    d_loss_real = self.discriminator.train_on_batch(y_vector, valid)
                    d_loss_fake = self.discriminator.train_on_batch(y_hat_vector, fake)
                    d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

                    d_loss_reals_tmp.append(d_loss_real[0])
                    d_loss_fakes_tmp.append(d_loss_fake[0])
                    d_losses_tmp.append(d_loss[0])





                    # ---------------------
                    #  Train Generator
                    # ---------------------

                    X = X_train[i * batchsize:(i + 1) * batchsize]
                    Y = X
                    y_true = y_train[i * batchsize :(i + 1) * batchsize ]  # y_T+1
                    y_true = np.expand_dims(y_true, 3)
                    g_in = X
                    y_vector = np.concatenate((Y, y_true), axis=1)

                    # Train the generator (to have the discriminator label samples as valid)
                    g_loss = self.combined.train_on_batch([g_in,Y], valid)
                    g_loss_prediction = self.generator.train_on_batch([g_in, Y], y_vector)

                    g_losses_tmp.append(g_loss)
                    g_loss_predictions_tmp.append(g_loss_prediction)

                    # Plot the progress
                    #if i%sample_interval==1:
                    print('       imf ',imf_index,', epoch ',epoch,' , batch ',i,'..............')
                    print ("    %d from %d batches, [D loss: %f, acc.: %.2f%%] [G loss: %f] [G loss prediction: %f]" %
                           (i,l, d_loss[0], 100*d_loss[1], g_loss,g_loss_prediction))
                t2=time.time()
                print('\t took : ',t2-t1)

                d_loss_reals.append(np.mean(d_loss_reals_tmp))
                d_loss_fakes.append(np.mean(d_loss_fakes_tmp))
                d_losses.append(np.mean(d_losses_tmp))
                g_losses.append(np.mean(g_losses_tmp))
                g_loss_predictions.append(np.mean(g_loss_predictions_tmp))


            trained_model=self.generator
            trained_model.save('g.h5')

            fig = plt.figure(facecolor='white', figsize=(10, 8))
            plt.subplot(321)
            plt.plot(d_loss_fakes,color='black',label='d_loss_fakes')
            plt.legend()
            plt.subplot(322)
            plt.plot(d_loss_reals, color='blue', label='d_loss_reals')
            plt.legend()
            plt.subplot(323)
            plt.plot(d_losses, color='red', label='d_losses')
            plt.legend()
            plt.subplot(324)
            plt.plot(g_losses, color='orange', label='g_losses')
            plt.legend()
            plt.subplot(325)
            plt.plot(g_loss_predictions, color='green', label='g_loss_predictions')
            plt.legend()
            plt.grid()
            plt.savefig('/home/vacek/Cloud/cloud-predictor/EPA-HTTP/prediction/GANS-EMD/10sec-smooth/resutls'
                        '/imf' + str(imf_index) + '/losses' + '.png', dpi=700)
            plt.pause(4)
            plt.close()

    def test(self):
        ytr=[]
        ypr=[]
        print('\n\t\t ------------------- Test staging ---------------------')
        for i in range(len(X_test)):
            X = X_test[i]
            X = np.expand_dims(X,3)
            X = np.reshape(X, newshape= (1,X.size,1))
            Y = X
            y_true = y_test[i]  # y_T+1
            y_true = np.expand_dims(y_true, 3)
            g_in = X
            print('\tshape of X is ', X.shape, 'shape of Y is ', Y.shape,
                  'shape of y_true is ', y_true.shape) if i==0 else None
            g_out = self.generator.predict([g_in, Y])  # which is y_hat_T+1
            print('\tshape of g_out is ',g_out.shape) if i==0 else None
            g_out=np.squeeze(g_out)
            y_pred=g_out[-1]
            ytr.append(y_true[0][0])
            ypr.append(y_pred)



        from sklearn.metrics import mean_squared_error
        from math import sqrt
        ms = mean_squared_error(ytr, ypr)
        rms = sqrt(ms)
        print('MSE is ', ms)
        print(' ------------------------------------')

        def mean_absolute_percentage_error(y_true, y_pred):
            ape = []
            for k in range(len(y_true)):
                # if abs(y_true[k])!=0 and k not in z1 and k not in z2:
                if abs(y_pred[k]) > 1e-3 and abs(y_true[k]) > 1e-3:
                    ape.append(abs((y_true[k] - y_pred[k]) / y_true[k]))
            # plt.hist(ape, bins='auto', color='orange')
            # plt.xlabel('MAPE')
            # plt.ylabel('frequency')
            # plt.grid()
            #
            # plt.pause(3)
            # plt.close()
            ape = sorted(ape)
            try:
                indexes = np.where(ape < np.percentile(ape, 90))[0]
                ape = [ape[k] for k in indexes]
                # print(ape)

                return np.mean(np.array(ape)) * 100
            except:
                return 0

        def median_absolute_percentage_error(y_true, y_pred):
            ape = []
            for k in range(len(y_true)):
                if abs(y_pred[k]) > 1e-3 and abs(y_true[k]) > 1e-3:
                    # if abs(y_true[k])!=0  and k not in z1 and k not in z2:
                    ape.append(abs((y_pred[k] - y_true[k]) / y_true[k]))
            ape = sorted(ape)
            try:
                indexes = np.where(ape < np.percentile(ape, 90))[0]
                ape = [ape[k] for k in indexes]
                return np.median(np.array(ape)) * 100
            except:
                return 0

        def mean_percentage_r_error(y_true, y_pred):
            ape = []
            for k in range(len(y_true)):
                if abs(y_pred[k]) > 1e-3 and abs(y_true[k]) > 1e-3:
                    # if abs(y_true[k])!=0  and k not in z1 and k not in z2:
                    ape.append(pow(((y_true[k] - y_pred[k]) / y_true[k]), 2))
            ape = sorted(ape)
            try:
                indexes = np.where(ape < np.percentile(ape, 90))[0]
                ape = [ape[k] for k in indexes]
                return sqrt(np.mean(np.array(ape)))
            except:
                return 0

        map = mean_absolute_percentage_error(ytr, ypr)
        print('MAPE is ', map)
        meap = median_absolute_percentage_error(ytr, ypr)
        print('MEAPE is ', meap)
        rmsre = mean_percentage_r_error(ytr, ypr)
        print('RMSRE is ', rmsre)

        # fig = plt.figure(facecolor='white', figsize=(10, 8))
        # plt.subplot(2, 1, 1)
        # plt.plot(ytr, label='Test Real Data', color='orange')
        # plt.legend()
        # plt.subplot(2, 1, 2)
        # plt.plot(ypr, label='Prediction Real Data, MAPE = %.4f%% ,\n '
        #                                     ' RMSE=%.4f  , RMSRE=%.4f  ' % (
        #                                     map, rms, rmsre))
        # plt.legend()
        # plt.savefig('/home/vacek/Cloud/cloud-predictor/NASA-HTTP/prediction/GANS/5min/resutls'
        #             '/imf' + str(imf_index) + '/prediction_normalize' + '.png', dpi=700)
        # plt.pause(5)
        # plt.close()

        ''' denoramalize the actual and predicted data '''
        min_test, max_test, ytr_revert = denorm_v2(ytr, MaxAbsScalerObj)
        min_predicted, max_predicted, ypr_revert = denorm_v2(ypr, MaxAbsScalerObj)


        ytr_revert = np.reshape(ytr_revert, (ytr_revert.size,))
        ypr_revert = np.reshape(ypr_revert, (ypr_revert.size,))

        map_denormalize = mean_absolute_percentage_error(ytr_revert, ypr_revert)
        print('MAPE in original scale is ', map_denormalize)
        meap_denormalize = median_absolute_percentage_error(ytr_revert, ypr_revert)
        print('MEAPE in original scale is ', meap_denormalize)
        rms_denormalize = sqrt(mean_squared_error(ytr_revert, ypr_revert))
        print('RMSE in original scale is ', rms_denormalize)
        rmsre_denorm = mean_percentage_r_error(ytr_revert, ypr_revert)
        print(' ------------------------------------')

        fig = plt.figure(facecolor='white', figsize=(10, 8))
        plt.subplot(3, 1, 1)
        plt.plot(ts_test ,ytr_revert, label='Test Real Data', color='orange')
        plt.plot(ts_test ,ypr_revert, label='Prediction Real Data', color='brown',alpha=0.5)
        plt.legend()
        plt.subplot(3, 1, 2)
        plt.plot(ts_test ,ytr_revert, label='Test Real Data', color='orange')
        plt.legend()
        plt.subplot(3, 1, 3)
        plt.plot(ts_test ,ypr_revert, label='Prediction Real Data, MAPE = %.4f%% ,\n '
                            ' RMSE=%.4f  , RMSRE=%.4f  ' % (
                                map_denormalize, rms_denormalize, rmsre_denorm))
        plt.legend()
        plt.savefig('/home/vacek/Cloud/cloud-predictor/EPA-HTTP/prediction/GANS-EMD/10sec-smooth/resutls'
                    '/imf' + str(imf_index) + '/prediction_original' + '.png', dpi=700)
        plt.pause(5)
        plt.close()

        print('writing to DB!')
        print(len(ts_test),len(ypr_revert),len(ytr_revert),len(X_test))
        for k in range(len(ts_test)):
            #print(ts_test[k], ytr_revert[k], ypr_revert[k],ytr[k])
            cur.execute('update epa_http_emd_10sec_copy set num_req_pred_gan=%s where imf_index=%s'
                        ' and num_req_pred is null and ts=%s', \
                        (float(ypr_revert[k]), int(imf_index), int(ts_test[k])+seq_len+1))
            conn.commit()


if __name__ == '__main__':
    seq_len = 30
    norm_version = 1  # v2= MinMaxScaler(0,1) , v1=MaxAbsScaler(-1,1)


    for imf_index in range(1,15):
        epoches=50 if imf_index<=4 else 10

        X_train, y_train, y_train_original_part, X_test, y_test, ts_train, ts_test, MaxAbsScalerObj = \
            Train_LSTM.load_data(seq_len, imf_index, norm_version)

        print(' --------------\n Shape of data is : \n ')
        print('X_train: ', X_train.shape, ' Y_train: ', y_train.shape)
        print('X_test: ', X_test.shape, ' Y_test: ', y_test.shape)
        print('----------------\n')

        gan = GAN()
        gan.train(epochs=epoches,batchsize=32,verbose=False)
        gan.test()

