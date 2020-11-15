from torch import argmax, from_numpy
import numpy as np
import cv2
from random import random

from cnn.structure import DroneQNet
from cnn.train import train_qnet
from gym_drone.envs.drone_env import DroneEnv
from constants import IMG_H, IMG_W, actions


def init_environment(map_file='map.csv', stations_file='bs.csv'):
    env = DroneEnv()

    # Get relevance map
    rel_map = np.genfromtxt(map_file, delimiter=';', dtype='int')
    print(rel_map)
    env.get_map(rel_map)

    # Get base stations
    base_stations = np.genfromtxt(stations_file, delimiter=';', dtype='int')
    print(base_stations)
    base_coord = env.get_bases(base_stations)
    print(base_coord)
    return env


def train_RL(episodes, iterations, env, action_epsilon, epsilon_decrease, batch_size):
    #    Initialization
    model = DroneQNet(2, IMG_W, IMG_H, len(actions))
    model.double()
    replay_memory = []
    ##

    iter_counts = 0
    for i in range(episodes):
        print("episode No",i)
        # the current state is the initial state
        state_matrix, cameraspot, _ = env.reset()
        cs = state_matrix, cv2.resize(env.get_part_relmap_by_camera(cameraspot), state_matrix.shape)
        done = False
        cnt = 0 # number of moves in an episode
        while not done:
            env.render()
            cnt += 1
            iter_counts += 1
            # select random action with eps probability or select action from model
            a = select_action(model, cs, action_epsilon)
            # update epsilon value taking into account the number of iterations
            #action_epsilon -= epsilon_decrease
            update_epsilon(action_epsilon, epsilon_decrease, iter_counts)

            observation, reward,done, _ = env.step(a)
            if done and cnt < 200:
                reward = -1000
            # TODO Alina must gave the same type of return in env.reset and the output of observation
            state_matrix, _, cameraspot = observation
            spot_rm = cv2.resize(env.get_part_relmap_by_camera(cameraspot), state_matrix.shape)
            replay_memory.append((state_matrix, a, spot_rm, reward))

            # training the model after batch_size iterations
            if iter_counts % batch_size == 0:
                data = np.random.permutation(replay_memory)[:batch_size]
                train_qnet(model, data)
            cs = state_matrix, spot_rm

        print ("Episode finished after {} timesteps".format(cnt))


def select_action(model, cs, action_epsilon):
    if random() > action_epsilon:
        x = from_numpy(np.stack(cs)).unsqueeze(dim=0)
        pred = model(x)

        position = argmax(pred, dim=1)
        return position.item()
    act = list(actions.keys())
    return np.random.choice(act)


def update_epsilon(action_epsilon, epsilon_decrease, iter_counts):
    # TODO do this properly
    return action_epsilon


if __name__ == '__main__':
    env = init_environment()
    train_RL(100, env.max_battery, env, 0.6, 0.01, 20)
    env.close() 