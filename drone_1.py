import numpy as np
import time

from gym_drone.envs.drone_env import DroneEnv


# alina
env = DroneEnv()

done = False
cnt = 0

observation, cameraspot, _ = env.reset()
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
    time.sleep(2)
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
