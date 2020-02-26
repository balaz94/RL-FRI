import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

data = []
colors = ['blue', 'red', 'green', 'black', 'pink', 'brown', 'yellow', 'indianred', 'fuchsia', 'orange', 'royalblue', 'aqua', 'grey', 'chartreuse']
names = ['pong/log_1_0_A2C.txt', 'pong/log_1_1_A2C.txt', 'pong/log_1_2_A2C.txt']


for i in range(len(names)):
    d = pd.read_csv(names[i])
    plt.plot(d['step'], d['score'], color=colors[i], label=names[i])

plt.legend(loc='upper left')
plt.show()

for i in range(len(names)):
    d = pd.read_csv(names[i])
    plt.plot(d['iteration'], d['score'], color=colors[i], label=names[i])

plt.legend(loc='upper left')
plt.show()
