import numpy as np
import Read_Data
import normalizer
from generator import generator

std_cpu, std_ram, mean_cpu, mean_ram, ts, \
cpu_values_normalize, \
ram_values_normalize=\
                 normalizer.normalizer(plot=True)


lookback = 50
step = 1
delay = 1
batch_size = 1


# lookback = 1440
# step = 6
# delay = 144
# batch_size = 128


# ----------- RAM ---------
data=np.vstack((ts,ram_values_normalize))
data=data.T
print('************',data.shape)



l=data.shape[0]
factor1=int(0.8*l)
factor2=int(0.9*l)

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

test_data=np.array(data[:,1])[factor2:]
print(test_data.shape)

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
model.add(layers.LSTM(64, activation='relu',
                      return_sequences=True))
model.add(layers.GRU(128, activation='relu',
                      dropout=0.1,
                      recurrent_dropout=0.3))
model.add(layers.Dense(32))
model.add(layers.Dense(1))



from keras.utils.vis_utils import plot_model
plot_model(model, to_file='model_plot.png', show_shapes=True, show_layer_names=True)

model.compile(optimizer=RMSprop(), loss='mae')
history = model.fit_generator(train_gen,
                              steps_per_epoch=100,
                              epochs=5,
                              validation_data=val_gen,
                              validation_steps=val_steps)

# import matplotlib.pyplot as plt
# loss = history.history['loss']
# val_loss = history.history['val_loss']
# epochs = range(1, len(loss) + 1)
# plt.figure()
# plt.plot(epochs, loss, 'bo', label='Training loss')
# plt.plot(epochs, val_loss, 'b',color='red', label='Validation loss')
# plt.title('Training and validation loss')
# plt.legend()
# plt.show()

prediction=model.predict_generator(test_gen,steps = test_steps)
print(prediction.shape)

import matplotlib.pyplot as plt
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




