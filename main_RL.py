import numpy as np

from cnn.structure import DroneQNet
from cnn.train import train_qnet
from gym_drone.envs.drone_env import DroneEnv
from constants import IMG_H, IMG_W, actions


def train_RL(episodes, iterations, base_stations, relevance_map, action_epsilon, epsilon_decrease, batch_size):
    #    Initialization
    model = DroneQNet(len(actions))
    replay_memory = []
    env = DroneEnv(base_stations)
    ##

    iter_counts = 0
    for i in range(episodes):
        # the current state is the initial state
        cs, _, _ = env.reset()

        for j in range(iterations):
            iter_counts += 1
            # select random action with eps probability or select action from model
            a = select_action(model, actions, action_epsilon)
            # update epsilon value taking into account the number of iterations
            action_epsilon = update_epsilon(action_epsilon, epsilon_decrease, iter_counts)

            observation, reward, _, _ = env.step(a)
            replay_memory.append((cs, a, observation, reward))

            if len(replay_memory) >= batch_size:
                data = np.random.choice(replay_memory, batch_size)
                train_qnet(model, data)
            cs = observation
