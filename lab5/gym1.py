import gym
import random
import time

from gym import envs
#print(envs.registry.all()) 
"""_summary_
customize env path: D:\Anaconda3\envs\pytorch_gpu\Lib\site-packages\gym\envs
"""

env = gym.make('BasicEnv-test')

env.reset() 
action_space = env.action_space
while True:
    action = random.choice(action_space) 
    state, reward, done, info = env.step(action)
    print('reward: %d' % reward)
    env.render()
    time.sleep(0.5)
    if done: break