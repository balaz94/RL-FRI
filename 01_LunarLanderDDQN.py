import gym
import datetime
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from agents.ddqn import AgentDDQN
from agents.experience_replay import ExperienceReplay

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()

        self.fc1 = nn.Linear(8, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 4)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

def learning():
    env = gym.make('LunarLander-v2')
    actions = 4
    state_dim = (8, )

    games_count = 1000
    scores = []

    experience_replay = ExperienceReplay(10000, state_dim)
    agent = AgentDDQN(0.99, actions, Net(), experience_replay, 0.001)

    d_now = datetime.datetime.now()
    for i in range(1, games_count):
        score = 0
        terminal = False
        state = env.reset()
        state = torch.from_numpy(state).double()

        while not terminal:
            action = agent.choose_action(state)
            state_, reward, terminal, _ = env.step(action)
            state_ = torch.from_numpy(state_).double()
            agent.store(state, action, reward, state_, terminal)
            agent.learn()
            state = state_
            score += reward

        scores.append(score)

        print('episode: ', i, '\t\tscore: ', + score, '\t\taverage score:' , np.average(scores[-100:]), '\t\tepsilon: ', agent.epsilon)
        if i % 10 == 0:
            d_end = datetime.datetime.now()
            d = d_end - d_now
            print('time: ', d)

if __name__ == '__main__':
    learning()
