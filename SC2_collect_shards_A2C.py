from collections import deque
from envs.sc2_shards_env import Env
import torch.nn as nn
import torch.nn.functional as F
import torch
import numpy as np
from agents.a2c import AgentA2C, Worker

MAX_SCORE_COUNT = 100


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, stride=2)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=2)
        self.conv3 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=2)

        self.fc1 = nn.Linear(7*7*32, 256)
        self.fc2 = nn.Linear(256, 16)
        self.fc3 = nn.Linear(7*7*32, 256)
        self.fc4 = nn.Linear(256, 1)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.view(-1, self.num_flat_features(x))
        x_logit = F.relu(self.fc1(x))
        logit = self.fc2(x_logit)
        x_value = F.relu(self.fc3(x))
        value = self.fc4(x_value)
        return logit, value

    def num_flat_features(self, x):
        size = x.size()[1:]
        num_features = 1
        for s in size:
            num_features *= s
        return num_features


def learning():
    actions = 16

    agent = AgentA2C(0.99, actions, Net(), 0.001, beta_entropy = 0.001, id=0, name='sc2_collect_shards')
    #agent.load_model()

    workers = []
    for id in range(10):
        env = Env()
        w = Worker(id, env, agent)
        workers.append(w)

    agent.learn(workers, 16, 200000)

def animate():
    scores = []
    avg_scores = deque([])
    actions = 16
    env = Env()

    agent = AgentA2C(0.99, actions, Net(), id=0, name='sc2_collect_shards')
    agent.load_model()

    i = 0
    while True:
        score = 0
        terminal = False
        state = env.reset()
        i += 1

        while not terminal:
            action = agent.choose_action(torch.from_numpy(state).double())
            state, reward, terminal, _ = env.step(action)
            score += reward

        avg_scores.append(score)
        if len(avg_scores) > MAX_SCORE_COUNT:
            avg_scores.popleft()

        scores.append(score)
        print('episode: ', i, '\t\tcumulative score:', round(np.average(scores), 3), '\t\tavg-100 score:',round(np.average(avg_scores), 3))

if __name__ == '__main__':
    #learning()
    animate()