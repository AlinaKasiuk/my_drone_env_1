import numpy as np
from PIL import Image
import time

from gym_drone.envs.drone_env import DroneEnv


# alina
env = DroneEnv()

done = False
cnt = 0

#Get relevance map
rel_map = np.genfromtxt('map.csv',delimiter=';',dtype='int')
print (rel_map)
env.get_map(rel_map)

#Get base stations
base_stations = np.genfromtxt('bs.csv',delimiter=';',dtype='int')
print (base_stations)
base_coord=env.get_bases(base_stations)
print (base_coord)


observation, cameraspot, square = env.reset()
print('Coordinates:')
print(observation)
print('Camera spot:')
print(cameraspot)
print('Camera spot square:')
print(square)

while not done:
    env.render()
    time.sleep(0.1)
    cnt += 1
    
    action = env.action_space.sample()
    print('Action:', action)
    observation, reward, done, _ = env.step(action)
    state_matrix, state, cameraspot=observation
    state_img = Image.fromarray(state_matrix, 'RGB')
    
    print('Timestep:', cnt)  
    print('State:', state )
    print('Camera spot:', cameraspot )    
    print('Reward:', reward)    
    if done:
        break
    
env.close()    
print ("Episode finished after {} timesteps".format(cnt))