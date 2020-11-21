import numpy as np
import pandas as pd


def build_1s(w, h, name="ones.csv"):
    map = np.ones((w, h))
    df = pd.DataFrame(map)
    df.to_csv(name, index=False, header=False, sep=";")


if __name__ == '__main__':
    build_1s(64, 64)
