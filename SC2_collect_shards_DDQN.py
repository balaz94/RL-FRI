from collections import deque
from envs.sc2_shards_env_dqn import Env
import datetime
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from agents.ddqn import AgentDDQN
from agents.experience_replay import ExperienceReplay

MAX_SCORE_COUNT = 100


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=1, stride=1)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1,padding=1)

        self.conv3 = nn.Conv2d(in_channels=32, out_channels=2, kernel_size=1, stride=1)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.conv3(x)
        x = x.view(-1, self.num_flat_features(x))  # flatten
        return x

    def num_flat_features(self, x):
        size = x.size()[1:]
        num_features = 1
        for s in size:
            num_features *= s
        return num_features


def learning(episode_count):
    env = Env()

    actions = 2*64*64
    input_dim = (3,64,64)

    scores = []
    avg_scores = deque([])

    gamma = 0.9
    lr = 0.0001
    exp_buffer_size = 20000
    batch_size = 256

    buffer =  ExperienceReplay(exp_buffer_size, input_dim)

    agent = AgentDDQN(gamma, actions, Net(), buffer,lr, update_steps = 1500, batch_size = batch_size, path="trained_models/sc2_collect_shards")

    agent.load_model()

    d_now = datetime.datetime.now()
    for i in range(0, episode_count):
        score = 0
        terminal = False
        state = env.reset()
        state = torch.from_numpy(state).double()

        while not terminal:
            action = agent.choose_action(state)
            state_, reward, terminal = env.step(action)
            state_ = torch.from_numpy(state_).double()
            score += reward
            agent.store(state, action, reward, state_, terminal)
            agent.learn()
            state = state_

        avg_scores.append(score)
        if len(avg_scores) > MAX_SCORE_COUNT:
            avg_scores.popleft()

        scores.append(score)

        print('episode: ', i, '\t\tscore: ', + round(score,3), '\t\tcumulative score:' , round(np.average(scores),3),'\t\tavg-100 score:' , round(np.average(avg_scores),3), '\t\tepsilon: ', round(agent.epsilon,4))
        if i % 100 == 0:
            d_end = datetime.datetime.now()
            d = d_end - d_now
            print('time: ', d)
            if i % 100 == 0  and i > 0:
                agent.save_model()


def animation():
    env = Env()

    actions = 2*64*64
    input_dim = (2,64,64)

    scores = []
    avg_scores = deque([])

    gamma = 0.9
    lr = 0.0001
    exp_buffer_size = 0
    batch_size = 0

    buffer = ExperienceReplay(exp_buffer_size, input_dim)

    agent = AgentDDQN(gamma, actions, Net(), buffer, lr, update_steps=1000, batch_size=batch_size,epsilon=0.01, path="trained_models/sc2_collect_shards")

    agent.load_model()

    i = 0
    while True:
        score = 0
        terminal = False
        state = env.reset()
        state = torch.from_numpy(state).double()
        i += 1

        while not terminal:
            action = agent.choose_action(state)
            state_, reward, terminal = env.step(action)
            state = torch.from_numpy(state_).double()
            score += reward

        avg_scores.append(score)
        if len(avg_scores) > MAX_SCORE_COUNT:
            avg_scores.popleft()

        scores.append(score)
        print('episode: ', i, '\t\tcumulative score:' , round(np.average(scores),3),'\t\tavg-100 score:' , round(np.average(avg_scores),3))


if __name__ == "__main__":
    #learning(100000)
    animation()