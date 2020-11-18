import torch
from torch.optim import Adam
from torch.nn import SmoothL1Loss
import numpy as np


def train_qnet(agent, data):
    total_loss = 0

    model_input = torch.from_numpy(data[:, 0])
    target_input = torch.from_numpy(data[:, 2])
    actions = data[:, 1]
    rewards = data[:, 3]
    dones = data[:, 4]

    y_hat = model(model_input)[np.arange(len(data)), actions]
    y_target = target_model(target_input).max(dim=1)[0]

    y_target[dones] = 0.0
    target_q_value = rewards + 0.8*y_target

    loss = criterion(target_q_value, y_hat)
    loss.backward()
    optimizer.step()

    # for cs, a, ns, reward, done in data:
    #     y_hat = model(model_input)
    #     y = torch.zeros(model.num_actions, dtype=torch.float64)
    #     y[a] = np.float(reward)
    #     loss = criterion(y_hat, y)
    #
    #
    #     total_loss += loss.item()
    #     optimizer.zero_grad()
    #     loss.backward()
    #     optimizer.step()
    if len(data):
        print("Loss {0}".format(total_loss/len(data)))
    else:
        print("No data")
