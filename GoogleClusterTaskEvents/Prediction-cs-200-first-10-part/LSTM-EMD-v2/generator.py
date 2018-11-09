import numpy as np

'''
data—The original array of floating-point data, which you normalized in listing 6.32.
 lookback—How many timesteps back the input data should go.
 delay—How many timesteps in the future the target should be.
 min_index and max_index—Indices in the data array that delimit which timesteps to draw from.
 This is useful for keeping a segment of the data for validation and another for testing.
 shuffle—Whether to shuffle the samples or draw them in chronological order.
 batch_size—The number of samples per batch.
 step—The period, in timesteps, at which you sample data. You’ll set it to 6 in
order to draw one data point every hour.
'''


def generator(data, lookback, delay, min_index, max_index,
              shuffle=False, batch_size=128, step=6):
    if max_index is None:
        max_index = len(data) - delay - 1
    i = min_index + lookback
    while 1:
        if shuffle:
            rows = np.random.randint(
                min_index + lookback, max_index, size=batch_size)
        else:
            if i + batch_size >= max_index:
                i = min_index + lookback
            rows = np.arange(i, min(i + batch_size, max_index))
            i += len(rows)

        samples = np.zeros((len(rows),
                           lookback // step,
                           data.shape[-1]))
        targets = np.zeros((len(rows),))
        for j, row in enumerate(rows):
            indices = range(rows[j] - lookback, rows[j], step)
            samples[j] = data[indices]
            targets[j] = data[rows[j] + delay][1]
        yield samples, targets