import numpy as np
import Read_Data
import normalizer
from generator import generator
from matplotlib import pyplot as plt

imf_index=10

# std_cpu, std_ram, mean_cpu, mean_ram, ts, ts_reload, \
# cpu_values_normalize, cpu_reloaded_normalize, \
# ram_values_normalize, ram_reloaded_normalize = \
#                  normalizer.normalizer(imf_index,plot=False)

std_cpu,std_ram,mean_cpu,mean_ram,ts,cpu_values_normalize,ram_values_normalize = \
            normalizer.normalizer(imf_index,plot=False)

#
# from sklearn.preprocessing import MinMaxScaler
# offline_scaler = MinMaxScaler()
# offline_scaler.fit(ram_values_normalize.reshape(-1, 1))
# ram_values_normalize=np.array(offline_scaler.transform(ram_values_normalize.reshape(-1, 1)))



#
# lookback = 100  # Observations will go back 10 days
# step = 5  #  Observations will be sampled at one data point per hour.
# delay = 20  # Targets will be 24 hours in the future.
# batch_size = 128

lookback = 360
step = 6
delay = 36
batch_size = 128



l=len(ram_values_normalize)
factor1=int(0.8*l)
factor2=int(0.9*l)
print(factor1,factor2,l)
print('******************************')

# ----------- RAM ---------
data=np.vstack((ts,ram_values_normalize))
data=data.T
print('************',data.shape)

mean = data[:factor1].mean(axis=0)
data -= mean
std = data[:factor1].std(axis=0)
data /= std


plt.plot(data[:,0], data[:,1])
plt.show()


# mean = data[:factor1].mean(axis=0)
# data -= mean
# std = data[:factor2].std(axis=0)
# data /= std


train_gen = generator(data,lookback=lookback,delay=delay,min_index=0,max_index=factor1,
                        shuffle=True,step=step,batch_size=batch_size)
val_gen = generator(data, lookback=lookback, delay=delay, min_index=factor1, max_index=factor2,
                        step=step, batch_size=batch_size)
test_gen = generator(data, lookback=lookback, delay=delay, min_index=factor2, max_index=None,
                        step=step,batch_size=batch_size)

print('train length is ',factor1,'validation length is ',factor2-factor1,'test length is ',len(data)-factor2)

print('-----------------------------------------------------------------------------------------------')

val_steps = (factor2 - factor1 - lookback) //batch_size #How many steps to draw from val_gen in order to see the entire validation set
print('validation step is ',val_steps)

test_steps = (l - factor2 - lookback)//batch_size  # How many steps to drawfrom test_gen in order to see the entire test set
print('test step is ',test_steps)



from keras.models import Sequential
from keras import layers
from keras.optimizers import RMSprop,Adam


model = Sequential()
model.add(layers.LSTM(32,
                     return_sequences=True,
                     input_shape=(None, data.shape[-1])))
model.add(layers.GRU(64, activation='relu',
                     dropout=0.1,
                     recurrent_dropout=0.3))
model.add(layers.Dense(10, activation='relu'))
model.add(layers.Dense(1))

from keras.utils.vis_utils import plot_model
plot_model(model, to_file='model_plot.png', show_shapes=True, show_layer_names=True)

model.compile(optimizer=RMSprop(), loss='mae')
history = model.fit_generator(train_gen,
                              steps_per_epoch=200,
                              epochs=5,
                              validation_data=val_gen,
                              validation_steps=val_steps)

import matplotlib.pyplot as plt
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(1, len(loss) + 1)
plt.figure()
plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'b',color='red', label='Validation loss')
plt.title('Training and validation loss')
plt.legend()
plt.show()

prediction=model.predict_generator(test_gen,steps = test_steps)
print(prediction.shape)

test_data=np.array(data[:,1])[l-prediction.shape[0]:]
print(test_data.shape)

plt.subplot(211)
plt.plot(prediction, label='Training loss')
plt.legend()
plt.subplot(212)
plt.plot(test_data, 'b',color='red', label='Validation loss')
plt.title('Training and validation loss')
plt.legend()
plt.show()

model.save('my_model.h5')  # creates a HDF5 file 'my_model.h5'
del model  # deletes the existing model




