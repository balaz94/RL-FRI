import gym
import datetime
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from agents.ddqn import AgentDDQN
from agents.experience_replay import ExperienceReplay
from utils.init import weights_init_xavier
from utils.stat import write_to_file
from utils.pong_wrapper import make_env

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()

        self.conv1 = nn.Conv2d(4, 32, 8, 4)
        self.conv2 = nn.Conv2d(32, 64, 4, 2)
        self.conv3 = nn.Conv2d(64, 64, 4, 2)

        self.fc1 = nn.Linear(576, 512)
        self.fc2 = nn.Linear(512, 6)

        self.apply(weights_init_xavier)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.view(-1, 576)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

class Net2(nn.Module):
    def __init__(self):
        super(Net2, self).__init__()

        self.conv1 = nn.Conv2d(4, 32, 8, 4)
        self.conv2 = nn.Conv2d(32, 64, 4, 2)
        self.conv3 = nn.Conv2d(64, 64, 4, 2)

        self.fc1 = nn.Linear(576, 512)
        self.fc2 = nn.Linear(512, 6)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.view(-1, 576)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

def learning(num):
    env = make_env('PongNoFrameskip-v4')
    actions = 6
    input_dim = (4, 80, 80)
    scores = []

    experience_replay = ExperienceReplay(25000, input_dim)
    agent = AgentDDQN(0.99, actions, Net(), experience_replay, 0.0001,
                      update_steps=1000, batch_size = 32, epsilon_min = 0.02, epsilon = 1.0,
                      epsilon_dec=1e-4)

    '''
    d_now = datetime.datetime.now()
    text = 'episode,score,time,step,epsilon'
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
    '''

    i = 0
    step = 0
    d_start = datetime.datetime.now()
    print('start ', d_start)
    d_end = datetime.datetime.now() + datetime.timedelta(minutes=180)
    text = 'episode,score,time,step,epsilon'

    while d_end > datetime.datetime.now():
        score = 0
        terminal = False
        state = env.reset()
        state = torch.from_numpy(state).double()
        i += 1

        while not terminal:
            step += 1
            action = agent.choose_action(state)
            state_, reward, terminal, _ = env.step(action)
            state_ = torch.from_numpy(state_).double()
            score += reward

            agent.store(state, action, reward, state_, terminal)
            agent.learn()
            state = state_

        scores.append(score)
        avg = np.average(scores[-100:])
        d = datetime.datetime.now() - d_start
        text += '\n' + str(i) + ',' + str(avg) + ',' + str(d) + ',' + str(step) + ',' + str(agent.epsilon)

    write_to_file(text, 'logs/pong/log_1_' + str(num) + '_DDQN.txt')
def learning2(num):
    env = make_env('PongNoFrameskip-v4')
    actions = 6
    input_dim = (4, 80, 80)
    scores = []

    experience_replay = ExperienceReplay(25000, input_dim)
    agent = AgentDDQN(0.99, actions, Net2(), experience_replay, 0.0001,
                      update_steps=1000, batch_size = 32, epsilon_min = 0.02, epsilon = 1.0,
                      epsilon_dec=1e-4)

    '''
    d_now = datetime.datetime.now()
    text = 'episode,score,time,step,epsilon'
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
    '''

    i = 0
    step = 0
    d_start = datetime.datetime.now()
    print('start ', d_start)
    d_end = datetime.datetime.now() + datetime.timedelta(minutes=180)
    text = 'episode,score,time,step,epsilon'

    while d_end > datetime.datetime.now():
        score = 0
        terminal = False
        state = env.reset()
        state = torch.from_numpy(state).double()
        i += 1

        while not terminal:
            step += 1
            action = agent.choose_action(state)
            state_, reward, terminal, _ = env.step(action)
            state_ = torch.from_numpy(state_).double()
            score += reward

            agent.store(state, action, reward, state_, terminal)
            agent.learn()
            state = state_

        scores.append(score)
        avg = np.average(scores[-100:])
        d = datetime.datetime.now() - d_start
        text += '\n' + str(i) + ',' + str(avg) + ',' + str(d) + ',' + str(step) + ',' + str(agent.epsilon)

    write_to_file(text, 'logs/pong/log_2_' + str(num) + '_DDQN.txt')

if __name__ == '__main__':
    for i in range(3):
        learning(i)
        learning2(i)
