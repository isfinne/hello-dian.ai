import copy
import logging
import itertools
import sys

import numpy as np
np.random.seed(0)
import pandas as pd
import gym
from gym.wrappers.atari_preprocessing import AtariPreprocessing
from gym.wrappers.frame_stack import FrameStack
import matplotlib.pyplot as plt
import torch
torch.manual_seed(0)
from torch import nn
from torch import optim

logging.basicConfig(level=logging.DEBUG,
        format='%(asctime)s [%(levelname)s] %(message)s',
        stream=sys.stdout, datefmt='%H:%M:%S')
#env = FrameStack(AtariPreprocessing(gym.make('PongNoFrameskip-v4')),
#        num_stack=4)
env = FrameStack(AtariPreprocessing(gym.make("Pong-ram-v4")),
        num_stack=4)
env.env.env.unwrapped.np_random.seed(0) # set seed for noops
env.env.env.unwrapped.unwrapped.seed(0) # set seed for AtariEnv
for key in vars(env):
    logging.info('%s: %s', key, vars(env)[key])
for key in vars(env.spec):
    logging.info('%s: %s', key, vars(env.spec)[key])
class DQNReplayer:
    def __init__(self, capacity):
        self.memory = pd.DataFrame(index=range(capacity),
                columns=['state', 'action', 'reward', 'next_state', 'done'])
        self.i = 0
        self.count = 0
        self.capacity = capacity

    def store(self, *args):
        self.memory.loc[self.i] = args
        self.i = (self.i + 1) % self.capacity
        self.count = min(self.count + 1, self.capacity)

    def sample(self, size):
        indices = np.random.choice(self.count, size=size)
        return (np.stack(self.memory.loc[indices, field]) for field in
                self.memory.columns)
#智能体
class CategoricalDQNAgent:
    def __init__(self, env):
        self.action_n = env.action_space.n
        self.gamma = 0.99
        self.epsilon = 1. # exploration
        
        self.replayer = DQNReplayer(capacity=100000)

        self.atom_count = 51
        self.atom_min = -10.
        self.atom_max = 10.
        self.atom_difference = (self.atom_max - self.atom_min) \
                / (self.atom_count - 1)
        self.atom_tensor = torch.linspace(self.atom_min, self.atom_max,
                self.atom_count)

        self.evaluate_net = nn.Sequential(
                nn.Conv2d(4, 32, kernel_size=8, stride=4), nn.ReLU(),
                nn.Conv2d(32, 64, kernel_size=4, stride=2), nn.ReLU(),
                nn.Conv2d(64, 64, kernel_size=3, stride=1), nn.ReLU(),
                nn.Flatten(),
                nn.Linear(3136, 512), nn.ReLU(inplace=True),
                nn.Linear(512, self.action_n * self.atom_count))
        self.target_net = copy.deepcopy(self.evaluate_net)
        self.optimizer = optim.Adam(self.evaluate_net.parameters(), lr=0.0001)

    def reset(self, mode=None):
        self.mode = mode
        if mode == 'train':
            self.trajectory = []

    def step(self, observation, reward, done):
        state_tensor = torch.as_tensor(observation,
                dtype=torch.float).unsqueeze(0)
        logit_tensor = self.evaluate_net(state_tensor).view(-1, self.action_n,
                self.atom_count)
        prob_tensor = logit_tensor.softmax(dim=-1)
        q_component_tensor = prob_tensor * self.atom_tensor
        q_tensor = q_component_tensor.mean(2)
        action_tensor = q_tensor.argmax(dim=1)
        actions = action_tensor.detach().numpy()
        action = actions[0]
        if self.mode == 'train':
            if np.random.rand() < self.epsilon:
                action = np.random.randint(0, self.action_n)
            
            self.trajectory += [observation, reward, done, action]
            if len(self.trajectory) >= 8:
                state, _, _, act, next_state, reward, done, _ = \
                        self.trajectory[-8:]
                self.replayer.store(state, act, reward, next_state, done)
            if self.replayer.count >= 1024 and self.replayer.count % 10 == 0:
                self.learn()
        return action

    def close(self):
        pass

    def update_net(self, target_net, evaluate_net, learning_rate=0.005):
        for target_param, evaluate_param in zip(
                target_net.parameters(), evaluate_net.parameters()):
            target_param.data.copy_(learning_rate * evaluate_param.data
                    + (1 - learning_rate) * target_param.data)

    def learn(self):
        # replay
        batch_size = 32
        states, actions, rewards, next_states, dones = \
                self.replayer.sample(batch_size)
        state_tensor = torch.as_tensor(states, dtype=torch.float)
        reward_tensor = torch.as_tensor(rewards, dtype=torch.float)
        done_tensor = torch.as_tensor(dones, dtype=torch.float)
        next_state_tensor = torch.as_tensor(next_states, dtype=torch.float)

        # compute target
        next_logit_tensor = self.target_net(next_state_tensor).view(-1,
                self.action_n, self.atom_count)
        next_prob_tensor = next_logit_tensor.softmax(dim=-1)
        next_q_tensor = (next_prob_tensor * self.atom_tensor).sum(2)
        next_action_tensor = next_q_tensor.argmax(dim=1)
        next_actions = next_action_tensor.detach().numpy()
        next_dist_tensor = next_prob_tensor[np.arange(batch_size),
                next_actions, :].unsqueeze(1)
        
        # project
        target_tensor = reward_tensor.reshape(batch_size, 1) + self.gamma \
                * self.atom_tensor.repeat(batch_size, 1) \
                * (1. - done_tensor).reshape(-1, 1)
        clipped_target_tensor = target_tensor.clamp(self.atom_min,
                self.atom_max)
        projection_tensor = (1. - (clipped_target_tensor.unsqueeze(1)
                - self.atom_tensor.view(1, -1, 1)).abs()
                / self.atom_difference).clamp(0, 1)
        projected_tensor = (projection_tensor * next_dist_tensor).sum(-1)

        logit_tensor = self.evaluate_net(state_tensor).view(-1, self.action_n,
                self.atom_count)
        all_q_prob_tensor = logit_tensor.softmax(dim=-1)
        q_prob_tensor = all_q_prob_tensor[range(batch_size), actions, :]

        cross_entropy_tensor = -torch.xlogy(projected_tensor, q_prob_tensor
                + 1e-8).sum(1)
        loss_tensor = cross_entropy_tensor.mean()
        self.optimizer.zero_grad()
        loss_tensor.backward()
        self.optimizer.step()

        self.update_net(self.target_net, self.evaluate_net)

        self.epsilon = max(self.epsilon - 1e-5, 0.05)



agent = CategoricalDQNAgent(env)
# 训练&测试
def play_episode(env, agent, max_episode_steps=None, mode=None, render=False):
    observation, reward, done = env.reset(), 0., False
    agent.reset(mode=mode)
    episode_reward, elapsed_steps = 0., 0
    while True:
        action = agent.step(observation, reward, done)
        if render:
            env.render()
        if done:
            break
        observation, reward, done, _ = env.step(action)
        episode_reward += reward
        elapsed_steps += 1
        if max_episode_steps and elapsed_steps >= max_episode_steps:
            break
    agent.close()
    return episode_reward, elapsed_steps


logging.info('==== train ====')
episode_rewards = []
for episode in itertools.count():
    episode_reward, elapsed_steps = play_episode(env, agent, mode='train', render=1)
    episode_rewards.append(episode_reward)
    logging.debug('train episode %d: reward = %.2f, steps = %d',
            episode, episode_reward, elapsed_steps)
    if np.mean(episode_rewards[-5:]) > 16.:
        break
plt.plot(episode_rewards)


logging.info('==== test ====')
episode_rewards = []
for episode in range(100):
    episode_reward, elapsed_steps = play_episode(env, agent, render=1)
    episode_rewards.append(episode_reward)
    logging.debug('test episode %d: reward = %.2f, steps = %d',
            episode, episode_reward, elapsed_steps)
logging.info('average episode reward = %.2f ± %.2f',
        np.mean(episode_rewards), np.std(episode_rewards))
