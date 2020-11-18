import torch
from torch.optim import Adam
from torch.nn import SmoothL1Loss
import numpy as np

from cnn.structure import DroneQNet
from constants import IMG_H, IMG_W


class BasicAgent:

    def __init__(self, actions):
        self.model = DroneQNet(2, IMG_W, IMG_H, len(actions))
        self.model.double()
        self.target_model = DroneQNet(2, IMG_W, IMG_H, len(actions))
        self.target_model.double()

        self.criterion = SmoothL1Loss()
        self.optimizer = Adam(self.model.parameters(), lr=0.001)

        self.gamma = 0.8
        self.train_iterations = 0

    def replace_target_network(self):
        self.target_model.load_state_dict(self.model.state_dict())

    def train(self, data):
        self.optimizer.zero_grad()

        current_states = np.array([i for i in data[:, 0]])
        next_states = np.array([i for i in data[:, 2]])
        model_input = torch.from_numpy(current_states)
        target_input = torch.from_numpy(next_states)
        actions_indexes = np.array(data[:, 1], dtype=np.int)
        rewards = np.array(data[:, 3], dtype=np.int)
        dones = np.array(data[:, 4], dtype=np.bool)

        y_hat = self.model(model_input)[np.arange(len(data)), actions_indexes]
        y_target = self.target_model(target_input).max(dim=1)[0]

        y_target[dones] = 0.0
        target_q_value = torch.from_numpy(rewards) + 0.8 * y_target

        loss = self.criterion(target_q_value, y_hat)
        loss.backward()
        self.optimizer.step()

        self.train_iterations += 1
