from __future__ import print_function, division

from keras.datasets import mnist
from keras.layers import Input, Dense, Reshape, Flatten, Dropout
from keras.layers import BatchNormalization, Activation, ZeroPadding2D
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import UpSampling2D, Conv2D
from keras.models import Sequential, Model
from keras.optimizers import Adam

import matplotlib.pyplot as plt

import sys

import numpy as np
import Train_LSTM




epochs = 100
seq_len = 10
imf_index=16
norm_version=1  # v2= MinMaxScaler(0,1) , v1=MaxAbsScaler(-1,1)

X_train, y_train,y_train_original_part, X_test, y_test,ts_train,ts_test,MaxAbsScalerObj =\
        Train_LSTM.load_data(seq_len,imf_index,norm_version)



rnn_model = Train_LSTM.build_model([1, seq_len, 20,1])

class GAN():
    def __init__(self):
        self.rnn_model = rnn_model

        self.img_rows = 28
        self.img_cols = 28
        self.channels = 1

        self.seq_len = 10
        self.img_shape = (self.img_rows, self.img_cols, self.channels)

        optimizer = Adam(0.0002, 0.5)



        # Build and compile the discriminator
        self.discriminator = self.build_discriminator()
        self.discriminator.compile(loss='binary_crossentropy',
            optimizer=optimizer,
            metrics=['accuracy'])

        # Build the generator
        self.generator = self.build_generator()

        # The generator takes x_train & y_train as input and generates two vectors
        # iterate one point by point base on a sequense length ( T = seq_len)
        X = X_train[:self.seq_len]
        Y = y_train[:self.seq_len]
        g_in = np.vstack(X, Y)
        g_out = self.generator(g_in)

        # For the combined model we will only train the generator
        self.discriminator.trainable = False

        # The discriminator takes generated images as input and determines validity
        validity = self.discriminator(g_out)

        # The combined model  (stacked generator and discriminator)
        # Trains the generator to fool the discriminator
        self.combined = Model(g_in, validity)
        self.combined.compile(loss='binary_crossentropy', optimizer=optimizer)


    def build_generator(self):
        # the process is iterating one point by point ...
        # input is X_train & Y_train
        # out put is y_hat_vector & y_bar_vector

        model =self.rnn_model
        model.summary()
        print('> Data Loaded. Compiling...')

        # rnn_model.fit(
        #     X_train,
        #     y_train,
        #     batch_size=64,
        #     nb_epoch=epochs,
        #     validation_split=0.1)

        g_in = Input(shape=(self.seq_len,2,))
        ########## assume X_test & y_test is one point and iterating over test part ####
        y_hat_predicted = Train_LSTM.predict_point_by_point(model, X_test)
        y_true=y_test

        #constitute and cancat two vercotr for input to descriminator

        y_hat_vector = np.concatenate(y_train,y_hat_predicted)
        y_bar_vector = np.concatenate(y_train, y_true)
        g_out = np.vstack(y_hat_vector,y_bar_vector) # input to discriminator


        return Model(g_in, g_out)

    def build_discriminator(self):

        model = Sequential()

        model.add(Flatten(input_shape=self.img_shape))
        model.add(Dense(512))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dense(256))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dense(1, activation='sigmoid'))
        model.summary()

        img = Input(shape=self.img_shape)
        validity = model(img)

        return Model(img, validity)

    def train(self, epochs, batch_size=128, sample_interval=50):



        # Adversarial ground truths
        valid = np.ones((batch_size, 1))
        fake = np.zeros((batch_size, 1))

        for epoch in range(epochs):

            # ---------------------
            #  Train Discriminator
            # ---------------------

            # iterate one point by point base on a sequense length ( T = seq_len)
            X=X_train[:self.seq_len]
            Y=y_train[:self.seq_len]
            g_in=np.vstack(X,Y)


            # Generate a batch of new images
            g_out = self.generator.predict(g_in) # is vstack of Y_hat and Y_bar
            Y_hat=g_out[:,0]
            y_bar=g_out[:,1]



            # Train the discriminator
            d_loss_real = self.discriminator.train_on_batch(y_bar, valid)
            d_loss_fake = self.discriminator.train_on_batch(Y_hat, fake)
            d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

            # ---------------------
            #  Train Generator
            # ---------------------

            X = X_train[:self.seq_len]
            Y = y_train[:self.seq_len]
            g_in = np.vstack(X, Y)

            # Train the generator (to have the discriminator label samples as valid)
            g_loss = self.combined.train_on_batch(g_in, valid)

            # Plot the progress
            print ("%d [D loss: %f, acc.: %.2f%%] [G loss: %f]" % (epoch, d_loss[0], 100*d_loss[1], g_loss))

            # If at save interval => save generated image samples
            if epoch % sample_interval == 0:
                self.sample_images(epoch)

    def sample_images(self, epoch):
        r, c = 5, 5
        noise = np.random.normal(0, 1, (r * c, self.latent_dim))
        gen_imgs = self.generator.predict(noise)

        # Rescale images 0 - 1
        gen_imgs = 0.5 * gen_imgs + 0.5

        fig, axs = plt.subplots(r, c)
        cnt = 0
        for i in range(r):
            for j in range(c):
                axs[i,j].imshow(gen_imgs[cnt, :,:,0], cmap='gray')
                axs[i,j].axis('off')
                cnt += 1
        fig.savefig("%d.png" % epoch)
        plt.close()


if __name__ == '__main__':
    gan = GAN()
    gan.train(epochs=30000, batch_size=64, sample_interval=200)
