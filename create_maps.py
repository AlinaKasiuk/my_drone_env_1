import numpy as np
import pandas as pd
from random import randint


def build_1s(w, h, name="ones.csv"):
    map = np.ones((w, h))
    df = pd.DataFrame(map)
    df.to_csv(name, index=False, header=False, sep=";")
    
    
def build_10s(w, h, name="tens.csv"):
    map = np.ones((w, h))*10
    df = pd.DataFrame(map)
    df.to_csv(name, index=False, header=False, sep=";")    


def build_checkerboard(w, h, c, name="checkerboard.csv"):
    map = np.zeros((w,h),dtype=int) 
    for i in range (w):
        for j in range (h):
            if (i%(2*c)==0):
                if (j%(2*c)==0):
                    map[i:(i+c),j:(j+c)]=1
                    map[(i+c):(i+2*c),(j+c):(j+2*c)]=1                    
    df = pd.DataFrame(map)
    df.to_csv(name, index=False, header=False, sep=";")


def build_random_obstacle_maps(w, h, obs_n, obs_max_radius, name="obstacles.csv", fill_value=10, obs_value=-100):
    rel_map = np.ones((w, h))*fill_value
    add_obstacles(rel_map, obs_n, obs_max_radius, obs_value=-100)
    df = pd.DataFrame(rel_map)
    df.to_csv(name, index=False, header=False, sep=";")


def add_obstacles(rel_map, obs_n, obs_max_radius, obs_value=-100):
    w, h = rel_map.shape
    for i in range(obs_n):
        x = randint(0, w)
        y = randint(0, h)
        radius = randint(0, obs_max_radius)
        rel_map[x - radius:x + radius + 1, y - radius:y + radius + 1] = obs_value


if __name__ == '__main__':
    # build_1s(32, 32)
    # build_10s(32, 32)
    # build_checkerboard(32, 32, 8)

    build_random_obstacle_maps(32, 32, 12, 4, fill_value=10, obs_value=-100)
