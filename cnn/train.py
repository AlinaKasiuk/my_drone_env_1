import torch
from torch.optim import Adam
from torch.nn import SmoothL1Loss
import numpy as np


def train_qnet(model, data):
    criterion = SmoothL1Loss()
    optimizer = Adam(model.parameters(), lr=0.001)

    total_loss = 0
    for cs, a, rel_map, reward in data:
        model_input = torch.from_numpy(np.stack((cs, rel_map))).unsqueeze(dim=0)

        y_hat = model(model_input)
        y = torch.zeros(model.num_actions, dtype=torch.float64)
        y[a] = np.float(reward)
        loss = criterion(y_hat, y)

        total_loss += loss.item()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    if len(data):
        print("Loss {0}".format(total_loss/len(data)))
    else:
        print("No data")
