import gym

env_dict = gym.envs.registration.registry.env_specs.copy()
for env in env_dict:
    if 'drone-v0' in env:
        print("Remove {} from registry".format(env))
        del gym.envs.registration.registry.env_specs[env]

import numpy as np

#gym_pull.pull('github.com/jnc96/drone-gym')

env = gym.make('drone-v0')

done = False
cnt = 0

observation, cameraspot, _= env.reset()
print('Coordinates:')
print(observation)
print('Camera spot:')
print(cameraspot)

#Get relevance map
r = np.genfromtxt('map.csv',delimiter=';',dtype='int')
print (r)
x,y=env.get_map(r)
print (x,y)

while not done:
    env.render()
    cnt += 1
    
    action = env.action_space.sample()
    print('Action:', action)
    observation, reward, done, _ = env.step(action)
    print('Timestep:', cnt)  
    print('Observation:', observation )
    print('Reward:', reward)    
    if done:
        break
    
print ("Episode finished after {} timesteps".format(cnt))
