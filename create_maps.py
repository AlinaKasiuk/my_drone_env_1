import numpy as np
import pandas as pd


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
    
    
if __name__ == '__main__':
    #build_1s(32, 32)
    build_10s(32, 32)
    #build_checkerboard(32, 32, 8)