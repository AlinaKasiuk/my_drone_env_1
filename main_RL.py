from torch import argmax, from_numpy
import numpy as np
import pandas as pd
import math
import cv2
from random import random

from cnn.basic_agent import BasicAgent
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


def train_RL(episodes, replace_iterations, env, action_epsilon, epsilon_decrease, batch_size):
    #    Initialization
    agent = BasicAgent(actions)
    replay_memory = []
    ##

    iter_counts = 0
    df = pd.DataFrame(columns=['Episode', 'Number of steps', 'Total reward'])
    df_actions = pd.DataFrame(columns=['Episode', 'Step', 'Action', 'Action type', 'Reward'])
    for i in range(episodes):
        print("episode No", i)
        # the current state is the initial state
        state_matrix, cameraspot, _ = env.reset()
        cs = get_current_state(state_matrix, cameraspot)
        done = False
        cnt = 0 # number of moves in an episode
        total_reward = 0
        while not done:
            env.render()
            cnt += 1
            iter_counts += 1
            # select random action with eps probability or select action from model
            a, a_type = select_action(agent.model, cs, action_epsilon)
            # update epsilon value taking into account the number of iterations
            action_epsilon = update_epsilon(action_epsilon, epsilon_decrease, iter_counts)
            observation, reward, done, _ = env.step(a)
            df_actions.loc[iter_counts] = {'Episode': i, 'Step': cnt, 'Action': a, 'Action type': a_type,'Reward': reward}
            total_reward += reward
            if done and cnt < 200:
                reward = -1000
            # TODO Alina must gave the same type of return in env.reset and the output of observation
            state_matrix, _, cameraspot = observation
            new_state = get_current_state(state_matrix, cameraspot)
            replay_memory.append((cs, a, new_state, reward, done))
            # training the model after batch_size iterations
            if iter_counts % batch_size == 0:
                data = np.random.permutation(replay_memory)[:batch_size]
                # train_qnet(model, data)
                agent.train(data)
                if agent.train_iterations % replace_iterations == 0:
                    agent.replace_target_network()
            cs = new_state
            
        df.loc[i]={'Episode': i, 'Number of steps': cnt, 'Total reward': total_reward}
        print("Total reward:", total_reward)
        print("Episode finished after {0} timesteps".format(cnt))
    return df, df_actions


def select_action(model, cs, action_epsilon):
    if random() > action_epsilon:
        x = from_numpy(np.stack(cs)).unsqueeze(dim=0)
        pred = model(x)
        act_type = 'Model'
        position = argmax(pred, dim=1)
        return position.item(), act_type
    act = list(actions.keys())
    act_type = 'Random'
    return np.random.choice(act), act_type


def update_epsilon(action_epsilon, epsilon_decrease, iter_counts):
    # TODO do this properly
    # action_epsilon = math.pow(0.9, iter_counts/100.0)
    return action_epsilon


def get_current_state(state_matrix, camera):
    state_matrix = cv2.resize(state_matrix, (32, 32)) / 100
    resize_camera = cv2.resize(env.get_part_relmap_by_camera(camera), state_matrix.shape)
    return np.stack((state_matrix, resize_camera))


if __name__ == '__main__':
    # PARAMS
    # episodes, iterations, env, action_epsilon, epsilon_decrease, batch_size
    env = init_environment()
    action_eps = 0.4

    batch_s = 10
    replace_iter = 20
    #
    table, table_actions = train_RL(200, replace_iter, env, action_eps, 0.01, batch_s)
    env.close() 
    table.to_csv('episodes.csv', sep=';', index = False, header=True)
    table_actions.to_csv ('actions.csv', sep=';', index = False, header=True)
