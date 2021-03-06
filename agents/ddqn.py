import copy
import numpy as np
from random import randrange
import torch
import torch.nn as nn
import torch.optim as optim

class AgentDDQN:
    def __init__(self, gamma, actions_count, model, experience_replay, lr,
                 update_steps = 1000, batch_size = 64, path = 'path',
                 epsilon=1.0, epsilon_dec = 1e-4, epsilon_min = 0.01):

        self.gamma = gamma
        self.actions_count = actions_count
        self.online_model = model
        self.target_model = copy.deepcopy(model)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print('device: ', self.device)
        self.online_model.to(self.device)
        self.target_model.to(self.device)
        for param in self.target_model.parameters():
            param.requires_grad = False
        self.mse = nn.MSELoss()
        self.experience_replay = experience_replay
        self.optimizer = optim.Adam(self.online_model.parameters(), lr=lr)
        self.update_steps = update_steps
        self.current_steps = 0
        self.batch_size = batch_size
        self.path = path
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_dec = epsilon_dec

    def choose_action(self, state):
        r = np.random.random()

        if r < self.epsilon:
            action = randrange(self.actions_count)
            return action
        else:
            state = state.unsqueeze(0).to(self.device).float()
            with torch.no_grad():
                actions = self.online_model(state)
            action = torch.argmax(actions).item()
            return action

    def store(self, state, action, reward, state_, terminal):
        self.experience_replay.store(state, action, reward, state_, terminal)

    def learn(self):
        if self.experience_replay.index < 600:
            return

        self.optimizer.zero_grad()
        states, actions, rewards, states_, terminals = self.experience_replay.sample(self.batch_size)

        with torch.no_grad():
            q_next = self.online_model(states_.to(self.device)).cpu()
            q_evaluated = self.target_model(states_.to(self.device)).cpu()

        q_y = self.online_model(states.to(self.device))
        q_target = q_y.detach().cpu()

        for i in range(0, len(states)):
            q_target[i, actions[i]] = rewards[i] + self.gamma * q_evaluated[i, torch.argmax(q_next[i]).item()] * (1 - terminals[i])

        loss = self.mse(q_y, q_target.to(self.device))
        loss.backward()
        self.optimizer.step()

        self.current_steps += 1
        if self.current_steps == self.update_steps:
            self.target_model.load_state_dict(self.online_model.state_dict())
            self.current_steps = 0

        if self.epsilon > self.epsilon_min:
            self.epsilon = max(self.epsilon - self.epsilon_dec, self.epsilon_min)

    def save_model(self):
        torch.save(self.online_model, self.path + "_online.pt")
        torch.save(self.target_model, self.path + "_target.pt")

    def load_model(self):
        if torch.cuda.is_available():
            self.online_model = torch.load(self.path + "_online.pt")
        else:
            self.online_model = torch.load(self.path + "_online.pt", map_location = torch.device('cpu'))
        self.online_model.eval()
        if torch.cuda.is_available():
            self.target_model = torch.load(self.path + "_target.pt")
        else:
            self.online_model = torch.load(self.path + "_target.pt", map_location=torch.device('cpu'))
        self.target_model.eval()