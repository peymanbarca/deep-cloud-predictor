import psycopg2
import numpy as np
import normalizer

std_cpu, std_ram, mean_cpu, mean_ram, ts, \
    cpu_values_normalize, \
    ram_values_normalize\
        =normalizer.normalizer(plot=False)

import matplotlib.pyplot as plt

fig = plt.figure(facecolor='white',figsize=(13.0, 8.0))
ax = fig.add_subplot(211)
ax.plot(ts, cpu_values_normalize,color='red', label='CPU',alpha=0.85)
plt.legend()
plt.grid()
plt.ylim([0,1])
plt.ylabel('Normalized CPU Request')
ax = fig.add_subplot(212)
ax.plot(ts, ram_values_normalize,color='blue', label='RAM',alpha=0.85)
plt.legend()
plt.grid()
plt.ylim([0,1])
plt.ylabel('Normalized RAM Request')
plt.xlabel('Time Symbol')
#
plt.savefig('REAL_200.png', format='png', dpi=800)
plt.show()






