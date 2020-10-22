import gym
import gym_pull

env_dict = gym.envs.registration.registry.env_specs.copy()
for env in env_dict:
    if 'drone-v0' in env:
        print("Remove {} from registry".format(env))
        del gym.envs.registration.registry.env_specs[env]

import gym_drone

import numpy as np

#gym_pull.pull('github.com/jnc96/drone-gym')

env = gym.make('drone-v0')

done = False
cnt = 0

observation = env.reset()

while not done:
    env.render()
    print(observation)
    cnt += 1
    
    action = env.action_space.sample()
    
    observation, reward, done, _ = env.step(action)
    
    if done:
        break
    
print ('game lasted', cnt, 'moves')