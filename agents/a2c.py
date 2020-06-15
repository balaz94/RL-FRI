import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from utils.stat import write_to_file

class AgentA2C:
    def __init__(self, gamma, actions_count, model, lr = 0.0001, beta_entropy = 0.001, value_loss_coef = 0.5, id=1, name='a2c'):
        self.gamma = gamma
        self.actions_count = actions_count
        self.model = model

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print('device: ', self.device)
        self.model.to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)

        self.beta_entropy = beta_entropy
        self.value_loss_coef = value_loss_coef

        self.average_score = []
        self.episodes = 0

        self.id = id
        self.name = name

    def choose_action(self, state):
        state = state.unsqueeze(0).to(self.device).float()
        with torch.no_grad():
            logits, _ = self.model(state)
        probs = F.softmax(logits, dim=-1)
        probs = probs.cpu()
        action = probs.multinomial(num_samples=1).detach()
        return action[0].item()

    def learn(self, workers, max_steps, max_iteration, write=True, start_iteration=0):
        self.load_model()
        len_workers = len(workers)
        observations = []
        for worker in workers:
            observations.append(torch.from_numpy(worker.reset()).float())
        observations = torch.stack(observations).to(self.device)

        self.average_score = []
        best_avg = -100
        self.episodes = 0

        text = text = 'iteration,episode,score,step'
        iter_step = max_steps * len_workers

        for iteration in range(start_iteration, max_iteration):
            mem_values = []
            mem_log_probs = []
            mem_entropies = []

            mem_rewards = []
            mem_non_terminals = []

            for step in range(max_steps):
                logits, values = self.model(observations)
                logits, values = logits.cpu(), values.cpu()

                probs = F.softmax(logits, dim=-1)
                log_probs = F.log_softmax(logits, dim=-1)
                entropies = (log_probs * probs).sum(1, keepdim=True)

                actions = probs.multinomial(num_samples=1).detach()
                log_probs_policy = log_probs.gather(1, actions)

                mem_values.append(values)
                mem_entropies.append(entropies)
                mem_log_probs.append(log_probs_policy)

                rewards = torch.zeros([len_workers, 1])
                non_terminals = torch.ones([len_workers, 1], dtype=torch.int8)
                observations = []

                for i in range(len_workers):
                    o, rewards[i, 0], t = workers[i].step(actions[i].item())
                    if t == True:
                        non_terminals[i, 0] = 0
                        o = workers[i].reset()
                    observations.append(torch.from_numpy(o).float())
                observations = torch.stack(observations).to(self.device)

                mem_rewards.append(rewards)
                mem_non_terminals.append(non_terminals)

            with torch.no_grad():
                _, R = self.model(observations)
                R = R.detach().cpu()

            value_loss = torch.zeros([len_workers, 1])
            policy_loss = torch.zeros([len_workers, 1])

            for step in reversed(range(max_steps)):
                R = mem_rewards[step] + self.gamma * R * mem_non_terminals[step]
                advantage = R - mem_values[step]
                value_loss += advantage**2
                policy_loss += - mem_log_probs[step] * advantage.detach() + self.beta_entropy * mem_entropies[step]

            policy_loss = policy_loss / max_steps
            value_loss = value_loss / max_steps

            self.optimizer.zero_grad()
            loss = policy_loss.mean() + self.value_loss_coef * value_loss.mean()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 0.1)
            self.optimizer.step()

            avg = np.average(self.average_score[-100:])
            if avg > best_avg:
                best_avg = avg
                print('saving model, best score is ', best_avg)
                torch.save(self.model.state_dict(), 'trained_models/' + self.name + '_' + str(self.id) + '_a2c.pt')

            if iteration % 25 == 0 and iteration > 0:
                print(iteration, '\tepisodes: ', self.episodes, '\taverage score: ', avg)
                if write:
                    text += '\n' + str(iteration) + ',' + str(self.episodes) + ',' + str(avg) + ',' + str(iter_step * iteration)

                    if iteration % 10000 == 0:
                        self.average_score = self.average_score[-100:]
                        write_to_file(text, 'logs/' + self.name + '_' + str(self.id) + '_' + str(iteration) + '_a2c.txt')
                        torch.save(self.model.state_dict(), 'trained_models/' + self.name + '_' + str(self.id) + '_' + str(iteration) + '_a2c.pt')

    def load_model(self):
        self.model.load_state_dict(torch.load('trained_models/' + self.name + '_' + str(self.id) + '_a2c.pt'))

    def save_model(self):
        torch.save(self.model.state_dict(), 'trained_models/' + self.name + '_' + str(self.id) + '_a2c.pt')

def reward_func(r):
    return r

class Worker:
    def __init__(self, id, env, agent, reward_function = reward_func, print_score = False):
        self.id = id
        self.env = env

        self.print_score = print_score
        self.episode = 1
        self.state = None
        self.score = 0
        self.agent = agent
        self.reward_function = reward_function

    def reset(self):
        if self.print_score and self.episode % 10 == 0:
            print('worker: ', self.id, '\tepisode: ', self.episode, '\tscore: ', self.score)
        self.agent.average_score.append(self.score)
        self.agent.episodes += 1
        self.state = self.env.reset()
        self.episode += 1
        self.score = 0
        return self.state

    def step(self, action):
        self.state, r, t, _ = self.env.step(action)
        self.score += r

        r = self.reward_function(r)

        return self.state, r, t
