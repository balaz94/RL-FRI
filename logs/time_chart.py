import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

data = []
colors = ['blue', 'red', 'green', 'black', 'pink', 'brown', 'yellow', 'indianred', 'fuchsia', 'orange', 'royalblue', 'aqua', 'grey', 'chartreuse']
names = ['pong/log_1_DDQN_Keras.txt', 'pong/log_1_DDQN_Keras2.txt', 'pong/log_1_0_DDQN.txt', 'pong/log_1_1_DDQN.txt', 'pong/log_1_2_DDQN.txt',
         'pong/log_2_0_DDQN.txt', 'pong/log_2_1_DDQN.txt', 'pong/log_2_2_DDQN.txt']


for i in range(len(names)):
    d = pd.read_csv(names[i])
    d['time'] = pd.to_datetime(d['time'])
    plt.plot(d['time'], d['score'], color=colors[i], label=names[i])

plt.legend(loc='upper left')
plt.show()
