import numpy as np
import normalizer
import time
import os
import warnings
from numpy import newaxis
from keras.layers.core import Dense, Activation, Dropout
from keras.layers.recurrent import LSTM,Recurrent,SimpleRNN
from keras.layers import Conv1D,MaxPooling1D,Bidirectional
from keras.models import Sequential
from keras.optimizers import RMSprop

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' #Hide messy TensorFlow warnings
warnings.filterwarnings("ignore") #Hide messy Numpy warnings


def build_model(layers):
    model = Sequential()

    # model.add(Conv1D(filters=32, kernel_size= 5, activation='relu',
    #                         input_shape=(None, layers[0])))
    # model.add(MaxPooling1D(3))
    # model.add(Conv1D(filters=32, kernel_size= 5, activation='relu'))

    model.add(Bidirectional(LSTM(
        input_shape=(layers[1], layers[0]),
        output_dim=layers[1],
        return_sequences=True,dropout=0.1,recurrent_dropout=0.5)))
    #model.add(Dropout(0.2))


    model.add(Bidirectional(LSTM(
        layers[2],
        return_sequences=False,dropout=0.1,recurrent_dropout=0.5)))
    #model.add(Dropout(0.2))

    # model.add(LSTM(
    #     layers[3],
    #     return_sequences=False))
    # model.add(Dropout(0.2))

    model.add(Dense(
        output_dim=layers[3],activation='relu'))

    model.add(Dense(
        output_dim=layers[4]))
    model.add(Activation("linear"))

    start = time.time()
    model.compile(loss="mse", optimizer=RMSprop(lr=1e-4)) # mae or mse??
    print("> Compilation Time : ", time.time() - start)
    return model

def load_data(seq_len):
    ts, num_req_normalize, minMaxScaler\
        =normalizer.normalizer(plot=True)


    sequence_length = seq_len + 1

    # %%%%%%%%%%%% Num Of Req %%%%%%%%%%%%%%%%%%%%%%%
    result = []
    for index in range(len(num_req_normalize) - sequence_length):  ## looks like a moving avg !!!
            result.append(num_req_normalize[index: index + sequence_length])

    result = np.array(result)
    print('------------')
    print('shape of sequense created is ', result.shape)  # (len(total_data)-seq_length)   *   seq_length
    print('length of each sequence is ', len(result[0]))
    print('------------')
    ts = ts[:result.shape[0] - 1]

    row = round(0.9 * result.shape[0])
    train = result[:int(row), :]

    np.random.shuffle(train)
    x_train = train[:, :-1]
    ts_train = ts[:int(row)]
    y_train = train[:, -1]
    x_test = result[int(row):, :-1]
    y_test = result[int(row):, -1]
    ts_test = ts[int(row)-1:]

    y_train_original_part = num_req_normalize[:int(row)]

    l = len(result)
    l_train = x_train.shape
    l_test = x_test.shape
    print('total data length is :',l)
    print('train set length : ',l_train)
    print('test set length :', l_test)

    x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1)) # make 3D
    x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))

    return [x_train, y_train,y_train_original_part, x_test, y_test,ts_train,ts_test,minMaxScaler]

def predict_point_by_point(model, data):
    #Predict each timestep given the last sequence of true data, in effect only predicting 1 step ahead each time
    predicted = model.predict(data)
    predicted = np.reshape(predicted, (predicted.size,))
    return predicted





