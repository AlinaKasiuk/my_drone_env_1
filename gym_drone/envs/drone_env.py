import gym
import logging
import math
import numpy as np
from gym import error, spaces, utils
from gym.utils import seeding


class DroneEnv(gym.Env):

    """
    ONE DRONE ENVIROMENT
    
    Description:
        A camera-equipped UAV can fly over the environment to be monitored
        to optimize the visual coverage of high-relevance areas. 
        
        The drone starts ****, and the goal is to adopt a patrolling strategy
        to opimize the PARTIAL coverage throught time.

    Source:
        This environment corresponds to the version of the drone patrolling
        problem described by Piciarelli and Foresti

    Observation:
     current x-pos, current y-pos, current z-pos,
     terrain angle (from horizontal axis)
     
     Let's assume that the map is of grid size WxH (5x5). Position of the drone is represented as (grid x index,
     grid y index), where (0,0) is the top left of the grid ((W-1,H-1)(4,4) is max value))
     z-pos is the hight (Z is maximum flying height);
     
        Type: Box(4)   
        Num     Observation               Min                     Max
        0       Current x-pos              0                    (W-1)=4
        1       Current y-pos              0                    (H-1)=4
        2       Current z-pos              0                    (Z-1)=4
        3       Terrain angle              0                      2*pi 

    Actions:
        Type: Discrete(8)
        Num   Action
        0     None
        1     MoveForward
        2     MoveBackward
        3     MoveLeft
        4     MoveRight      
        5     MoveUp
        6     MoveDown    
        7     RotateLeft
        8     RotateRight         
python drone_1.py
    Reward:
    ***    

    Starting State:
    ***
        All observations are assigned a uniform random value in [1..2]

    Episode Termination:
    ***
        Episode length is greater than 200.
    """
    
    
    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second': 50
    }

    # TODO en el reset, escoger estado inicial aleatoriamente de las base stations
    def __init__(self, base_stations):
        self.bs = base_stations
        # debug vars
    
        self.__version__ = "2.1.1"
    
        # Hyperparameter definition 
        self.x_min = int(0)
        self.x_max = int(4)
        self.y_min = int(0)
        self.y_max = int(4)
        self.z_min = int(0)
        self.z_max = int(50)        
        self.min_terr_angle = 0
        self.max_terr_angle = 2*math.pi #terrain angle - something that is observed
        
        self.cam_angle = 0.25*math.pi
 
    
        self.delta_pos=1
        self.delta_angle=0.25*math.pi
        # ???
        self.state = None #initiate state holder
        self.cameraspot=None
        
        self.episode_over = False
        self.current_episode = -1 
        self.current_timestep = 0 # -1 because timestep increments before action
        self.current_pos = [0,0]
        self.action_episode_memory = []
        self.grid_step_max = (self.x_max+1)*(self.y_max+1) - 1 # number of grid squares
        self.max_timestep = 2*self.grid_step_max   # Visits all grid squares twice.
    
        # Observations are (in this order): current x-pos, current y-pos, terrain angle (from horizontal axis)
        # Let's assume that the map is of grid size 5x5. Position of the drone is represented as (grid x index,
        # grid y index), where (0,0) is the top left of the grid ((4,4) is max value)).
    
        # Here, low is the lower limit of observation range, and high is the higher limit.
        low_ob = np.array([self.x_min,  # x-pos
                           self.y_min,  # y-pos
                           self.min_terr_angle]) # terrain_angle_deg
        high_ob = np.array([self.x_max,  # x-pos
                            self.y_max,  # y-pos
                            self.max_terr_angle]) # terrain_angle_deg
        self.observation_space = spaces.Box(low_ob, high_ob, dtype=np.float32)
    
        self.action_space = spaces.Discrete(9) 
   
        # generate random terrain gradients/create them here
        # import random
        # list  = [111,222,333,444,555]
        # print("random.choice() to select random item from list - ", random.choice(list))

    
        self.relevance_map =   [[0,0,1,0,0],
                                [0,1,2,1,0],
                                [1,2,3,2,1],
                                [0,1,2,1,0],
                                [0,0,1,0,0]]
                           
        self.seed()
        self.viewer = None
        self.state = None
    
        self.steps_beyond_done = None                       


    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def _get_cameraspot(self):
        
        x, y, z, terr_angle = self.state
        terr_angle=0
        c=z
        #*math.tan(self.cam_angle))
        if  x-c<self.x_min:
            x_cam_min=self.x_min
        else:
            x_cam_min=x-c
            
        if  x+c>self.x_max:
            x_cam_max=self.x_max
        else:
            x_cam_max=x+c   
             
        if  y-c<self.y_min:
            y_cam_min=self.y_min
        else:
            y_cam_min=y-c
            
        if  y+c>self.y_max:
            y_cam_max=self.y_max
        else:
            y_cam_max=y+c


        p1=[x_cam_min,y_cam_max]
        p2=[x_cam_min,y_cam_min]
        p3=[x_cam_max,y_cam_min]
        p4=[x_cam_max,y_cam_max]
                
        tmp_cameraspot=[p1,p2,p3,p4]

        return tmp_cameraspot    
    
    
    def step(self, action):
        
        err_msg = "%r (%s) invalid" % (action, type(action))
        assert self.action_space.contains(action), err_msg    
    
        x, y, z, terr_angle = self.state
        terr_angle=0
        
        if terr_angle>2*math.pi:
            terr_angle=terr_angle-2*math.pi
        elif terr_angle<0:
            terr_angle=terr_angle+2*math.pi    
    
        done = bool(
            x < self.x_min
            or x > self.x_max
            or y < self.y_min
            or y > self.y_max
            or z < self.z_min
            or z > self.z_max
            )  
            
        if done:
            action=0
        else:    
            if action==1:
                alpha=terr_angle
                x=x+self.delta_pos*math.cos(alpha)
                y=y+self.delta_pos*math.sin(alpha)
            elif action==2:
                alpha=terr_angle+math.pi
                x=x+self.delta_pos*math.cos(alpha)    
                y=y+self.delta_pos*math.sin(alpha)
            elif action==3:
                alpha=terr_angle+0.5*math.pi
                x=x+self.delta_pos*math.cos(alpha)    
                y=y+self.delta_pos*math.sin(alpha)        
            elif action==4:
                alpha=terr_angle-0.5*math.pi
                x=x+self.delta_pos*math.cos(alpha)    
                y=y+self.delta_pos*math.sin(alpha) 
            elif action==5:
                z=z+self.delta_pos    
            elif action==6:
                z=z-self.delta_pos
            elif action==7:
                terr_angle=terr_angle-self.delta_angle    
            elif action==8:
                terr_angle=terr_angle+self.delta_angle
        
        self.state=(x, y, z, terr_angle) 
        self.cameraspot = self._get_cameraspot()
        
        self.observation=(self.state,self.cameraspot)
        reward=self.relevance_map[int(x)][int(y)]
      
      
        return np.array(self.observation), reward, done, {}

    def get_map(self,rel_map):
        self.relevance_map=rel_map
        ncol, nrow = rel_map.shape
        self.x_max=ncol-1
        self.y_max=nrow-1
        return self.x_max,self.y_max
        

    def reset(self):
        self.state = self.np_random.randint(low=10, high=15, size=(4,))
        self.cameraspot = self._get_cameraspot()
        self.steps_beyond_done = None
        return np.array(self.state),  np.array(self.cameraspot), {} 


    def render(self, mode='human'):
        screen_width = 500
        screen_height = 500

        world_width = self.x_max- self.x_min
        scale = screen_width/world_width
    
    
        drone_width = 10.0
        drone_len = 50.0
    
        if self.viewer is None:
            from gym.envs.classic_control import rendering
            
            self.viewer = rendering.Viewer(screen_width, screen_height)
            l, r, t, b = -drone_width / 2, drone_width / 2, drone_len / 2, -drone_len / 2
            axleoffset = drone_width / 4.0
            drone = rendering.FilledPolygon([(l, b), (l, t), (r, t), (r, b)])
            self.drone_trans = rendering.Transform()
            drone.add_attr(self.drone_trans)
            self.viewer.add_geom(drone)
            self.axle = rendering.make_circle(drone_width/2)
            self.axle.add_attr(self.drone_trans)
            self.axle.set_color(.5, .5, .8)
            self.viewer.add_geom(self.axle)

        if self.state is None:
            return None

        # Edit the pole polygon vertex
        x = self.state
        dronex = x[0] * scale 
        droney = x[1] * scale         
        self.drone_trans.set_translation(dronex, droney)
        self.drone_trans.set_rotation(x[3])

        return self.viewer.render(return_rgb_array=mode == 'rgb_array')


    def close(self):
        if self.viewer:
            self.viewer.close()
            self.viewer = None
    
        



