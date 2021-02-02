import numpy as np
import pandas as pd
from random import randint

class Map:

    def __init__(self, wigth, height, name="map.csv"):
        self.h = height
        self.w = wigth

        self.rel_map = np.zeros((wigth, height))
        self.file_name=name
        self.maptypes= {"Filled", "Checkerboard", "Random"}        
            
    def safe_to_csv(self):
        df = pd.DataFrame(self.rel_map)
        df.to_csv(self.file_name, index=False, header=False, sep=";")        
       
    def filled_map(self, value):
        self.rel_map = np.ones((self.w, self.h))*value
        return self.rel_map       
        
    def checkerboard_map(self, max_val, min_val, step):
        self.rel_map = np.ones((self.w, self.h))*self.min_val    
        for i in range (self.w):
            for j in range (self.h):
                if (i%(2*step)==0):
                    if (j%(2*step)==0):
                        self.rel_map[i:(i+step),j:(j+step)]=max_val
                        self.rel_map[(i+step):(i+2*step),(j+step):(j+2*step)]=max_val                       
        return self.rel_map 
    
    def random_map(self, max_val, min_val):    
        for i in range (self.w):
            for j in range (self.h):
                self.rel_map[i,j]=randint(min_val ,max_val)                       
        return self.rel_map 

    def add_obstacles(self, obs_value, obs_n, obs_max_radius):
        for i in range(obs_n):
            x = randint(0, self.w)
            y = randint(0, self.h)
            radius = randint(0, obs_max_radius)
            self.rel_map[x - radius:x + radius + 1, y - radius:y + radius + 1] = obs_value    
            
        
    def map_reset(self, maptype="Filled", max_value=10, min_value=0, checker_step=8, obstracles=False, obs_n=10, obs_max_radius=4, obs_value=-100):
        self.max=max_value
        self.min=min_value
        
        if maptype in self.maptypes:
            if maptype=="Filled":
                self.filled_map(max_value)
            if maptype=="Checkerboard":
                self.checkerboard_map(max_value, min_value, checker_step)
            if maptype=="Random":
                self.random_map(max_value, min_value)
        else:
            raise ValueError("The maptype is incorrect")     
        
        if obstracles:
            self.add_obstacles(obs_value, obs_n, obs_max_radius)
            
        self.safe_to_csv()
        
        return self.rel_map                  
                
    
    