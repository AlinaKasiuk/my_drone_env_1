import gym
import numpy as np

from gym import spaces
from gym.utils import seeding
import matplotlib.pyplot as plt
from gym_drone.envs import rendering


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
    
    Num |    Observation    |   Min   |    Max
    ----|-------------------|---------|-----------
    0   |    Current x-pos  |    0    |    W-1
    1   |    Current y-pos  |    0    |    H-1
    2   |    Current z-pos  |    0    |    Z-1
    4   |    Battery level  |    0    |     1

    ### Actions:

    Type: Discrete(10)

    Num  |  Action
    -----|------------------
    0    |  Forward
    1    |  Backward
    2    |  Left
    3    |  Right
    4    |  Up      
    5    |  Down   
    6    |  Forward and Left
    7    |  Forward and Right 
    8    |  Backward and Left
    9    |  Backward and Right  
       
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

    def __init__(self):
    
        # debug vars
    
        self.__version__ = "0.0.1"
    
        # Hyperparameter definition 
        # The borders for state values (changes with downloading a map)
        self.x_min = int(0)
        self.x_max = int(31)
        self.y_min = int(0)
        self.y_max = int(31)
        self.z_min = int(0)
        self.z_max = int(5)
        
        # k quality coeficient
        # TODO: define k with a formula
        self.k = {0: 0.5, 1: 1, 2: .875, 3: .75, 4:.625, 5: 0.5}

        self.min_battery = 0
        self.max_battery = 200
        
        # One step size
        self.delta_pos = 1
        self.delta_battery = 1

        # Initial values
        self.state = None   #initiate state holder
        self.cameraspot = None
        
        self.episode_over = False
        self.current_episode = -1 
        self.current_timestep = 0   # -1 because timestep increments before action
        self.current_pos = [0, 0]
        self.action_episode_memory = []
        self.grid_step_max = (self.x_max+1)*(self.y_max+1) - 1 # number of grid squares
        self.max_timestep = 2*self.grid_step_max   # Visits all grid squares twice.
    
        """
        # Observations are (in this order): current x-pos, current y-pos, terrain angle (from horizontal axis)
        # Let's assume that the map is of grid size 64x64. Position of the drone is represented as (grid x index,
        # grid y index), where (0,0) is the top left of the grid ((63,63) is max value)).
        """    
    
        # TODO: check observation_space. Battery - ?
        # Muy be it should be changed after changing the map
        
        # Here, low is the lower limit of observation range, and high is the higher limit.
        
        low_ob = np.array([self.x_min,  # x-pos
                           self.y_min,  # y-pos
                           self.z_min]) # z-pos
        high_ob = np.array([self.x_max,  # x-pos
                            self.y_max,  # y-pos
                            self.z_max]) # z-pos
        self.observation_space = spaces.Box(low_ob, high_ob, dtype=np.float32)
    
        self.action_space = spaces.Discrete(10) 
    
        # TODO: Is it needed to define the default relevance map?
        
        self.relevance_map = None
        self.initial_rm = None

        self.seed()
        self.viewer = None
        self.state = None
        self.state_matrix = None
        
    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def get_map(self, rel_map):
        "Importing the relevance map to the enviroment"
        self.relevance_map = np.array(rel_map)    # NumPy array
        self.initial_rm = np.array(rel_map)   # The initial map doesn't change every step
        ncol, nrow = rel_map.shape  # Map size
        self.x_max = ncol-1
        self.y_max = nrow-1
        
    def get_bases(self, bs):
        " Importing the base stations map to the environment "
        self.base_stations = bs     # NumPy array
        ncol, nrow = bs.shape
        
        # Getting a list of base station coordinates:
        x=[]
        y=[] 
        for i in range(0,ncol-1):
            for j in range(0,nrow-1):
                if bs[i][j]==100:
                    x.append(i)
                    y.append(j)
        self.base_x=np.array(x)
        self.base_y=np.array(y)
        Coord=np.array([self.base_x,self.base_y])
        self.base_coord=Coord.T
        
        return self.base_coord    
           
    def reset(self):
        "Reseting the enviroment: initial map, start position"
        
        # get_map(_) and get_bases(_) methods should be previously done:
        self._check_map() # check if the map is correctly downloaded
        self._check_bases() # check if the bases are correctly downloaded   
        
        # Start from one of the base stations:
        m,n = self.base_coord.shape
        i = np.random.randint(0, m-1) if m > 1 else 0
        x,y=self.base_coord[i]
        
        # The initial state:
        self.state = [x, y, 1, 100]
        self.cameraspot = self._get_cameraspot()
        
        # Define the initial relevance map
        self.relevance_map = np.array(self.initial_rm)
            
        return self.state_matrix,  np.array(self.cameraspot)   
    
    def step(self, action):
        "One state-action step"
        
        # Check if the selected action is in the set of actions: 
        err_msg = "%r (%s) invalid" % (action, type(action))
        assert self.action_space.contains(action), err_msg    
    
        # Saving the previous state:
        old_state = self.state        
        old_camera_spot = self._get_cameraspot()

        # Actualize the state with a new action
        new_state = self._take_action(action)
        new_cameraspot = self._get_cameraspot()
        
        # Check if the agent inside the map limits and it has some battery
        done = self._check_limits()
        
        # Observation tuple for output:
        self.observation = self.state_matrix, self.state, self.cameraspot
        
        # Calculating the reward funcion:
        reward = self._get_reward(old_state, new_state, old_camera_spot, new_cameraspot)
        
        # Zeroing the observed values:
        self._zero_rel_map(old_camera_spot)
        
        return self.observation, reward, done, {}
    
    def render(self, mode='human', show=True):
        if not self.state:
            return
        screen_width = 640
        screen_height = 640

        world_width = self.x_max - self.x_min
        scale = screen_width/world_width
    
        x = self.state
        dronex = x[0] * scale
        droney = x[1] * scale
        dronez = (x[2]*2+1) * scale
        red = (100-x[3])/100
        green = x[3]/100
        blue = 0
    
        drone_width = 40.0
        drone_len = 40.0

        if self.viewer is None:
            self.viewer = rendering.Viewer(screen_width, screen_height)

            l, r, t, b = -drone_width / 2, drone_width / 2, drone_len / 2, -drone_len / 2
            drone = rendering.FilledPolygon([(l, b), (l, t), (r, t), (r, b)])
            self.drone_trans = rendering.Transform()
            self.map_trans = rendering.Transform()
            self.drone_color = drone.attrs[0]
            drone.add_attr(self.drone_trans)
            self.map_trans.set_translation(screen_width/2, screen_height/2)

            map_image = self._get_map_image()
            map_img = rendering.Image(map_image, screen_width, screen_height)
            map_img.add_attr(self.map_trans)
            self.viewer.add_geom(map_img)

            self.viewer.add_geom(drone)
            self.axle = rendering.make_circle(5)
            self.axle.add_attr(self.drone_trans)
            self.axle.set_color(0, 0, 0)
            self.viewer.add_geom(self.axle)

        if self.viewer is not None and show:
            map_image = self._get_map_image()
            map_img = rendering.Image(map_image, screen_width, screen_height)
            map_img.add_attr(self.map_trans)
            self.viewer.geoms[0] = map_img

        if self.state is None:
            return None

        # Edit the pole polygon vertex
        self.drone_color.vec4 = ((red, green, blue, 1.0))
        self.drone_trans.set_translation(dronex, droney)
        self.drone_trans.set_scale(dronez/50, dronez/50)
        return self.viewer.render(return_rgb_array=mode == 'rgb_array')  
    
    def close(self):
        if self.viewer:
            self.viewer.close()
            self.viewer = None    

    def get_coverage_rate(self):
        "Part of the map already covered in percents"
        coverage_rate=100-self.relevance_map.sum()*100/512
        # TODO: Change. This is correct only for 0&1 
        return coverage_rate            
    
    def get_part_relmap_by_camera(self, camera_spot):
        camera_spot = np.array(camera_spot)
        x_min = camera_spot[:, 0].min()
        x_max = camera_spot[:, 0].max() + 1

        y_min = camera_spot[:, 1].min()
        y_max = camera_spot[:, 1].max() + 1

        # out of boundaries
        if x_min < 0 or y_min < 0 or x_max >= self.relevance_map.shape[0] or y_max >= self.relevance_map.shape[1]:
            return np.array([[-100.0]])
        return np.array(self.relevance_map[x_min:x_max, y_min:y_max], dtype=np.float)
         
    def _get_cameraspot(self):
        "Calculating the field of view (FOV), the state in matrix format"
        
        x, y, z, battery = self.state
        c=z # *math.tan(self.cam_angle))  

        x_cam_min = x-c
        x_cam_max = x+c
             
        y_cam_min = y-c
        y_cam_max = y+c

        # the coordinates of angles of FOV:
        p1=[x_cam_min,y_cam_max]
        p2=[x_cam_min,y_cam_min]
        p3=[x_cam_max,y_cam_min]
        p4=[x_cam_max,y_cam_max]
        tmp_cameraspot=[p1,p2,p3,p4] 

        ncol=self.x_max+1
        nrow=self.y_max+1
        
        # State to matrix:
        cam_matrix =np.zeros((ncol,nrow))
        cam_matrix[x_cam_min:x_cam_max+1,y_cam_min:y_cam_max+1]=battery
        self.state_matrix = cam_matrix
                
        return tmp_cameraspot

    def _get_distance(self, state):
        "Calculating the shortest distance from the current state to the base"
        min_dist = len(self.relevance_map)
        x, y, _, _ = state
        for bx, by in zip(self.base_x, self.base_y):
            dif_x = abs(x - bx)
            dif_y = abs(y - by)

            dist = max(dif_x, dif_y)
            if dist < min_dist:
                min_dist = dist
        return min_dist
    
    def _get_reward(self, old_state, new_state, old_camera_spot, new_camera_spot):
        "The reward formula implementation"
        
        x, y, z, battery = old_state
        dist = self._get_distance(new_state)        
        n_r = self.get_part_relmap_by_camera(new_camera_spot).sum()
        
       # if n_r < 0:
          #  print("EL nuevo estado esta fuera")
          #  self.relevance_map = np.array(self.initial_rm)
        #    return -100
        _, _, new_z, _ = new_state
        if new_z not in self.k:
         #   self.relevance_map = np.array(self.initial_rm)
          #  print("Altura no apropiada")
            return -100

        p = battery - dist < 0
        if p:
            #print("Too far from base station {0}".format(self._get_distance(old_state) - dist))
            return 60 * (self._get_distance(old_state) - dist)
        else:
            c_r = self.get_part_relmap_by_camera(old_camera_spot).sum()
            reward = self.k[new_z] * (n_r - c_r)
            return reward

    def _zero_rel_map(self, camera_spot):
        camera_spot = np.array(camera_spot)
        x_min = camera_spot[:, 0].min()
        x_max = camera_spot[:, 0].max() + 1

        y_min = camera_spot[:, 1].min()
        y_max = camera_spot[:, 1].max() + 1

        if x_min >= 0 and y_min >= 0:
            self.relevance_map[x_min:x_max, y_min:y_max] -=1

    def _take_action(self, action):
        x, y, z, battery = self.state
        
        battery -= self.delta_battery
        
        if action == 0:
            y += self.delta_pos   # Forward
        elif action == 1:
            y -= self.delta_pos   # Backward
        elif action==2:
            x-=self.delta_pos   # Left          
        elif action==3:
            x+=self.delta_pos   # Right   
        elif action==4:
            z+=self.delta_pos   # Up   
        elif action==5:
            z-=self.delta_pos   # Down   
        elif action==6:
            y+=self.delta_pos   # Forward            
            x-=self.delta_pos   # Left             
        elif action==7:
            y+=self.delta_pos   # Forward            
            x+=self.delta_pos   # Left  
        elif action==8:
            y-=self.delta_pos   # Backward             
            x-=self.delta_pos   # Left             
        elif action==9:
            y-=self.delta_pos   # Backward           
            x+=self.delta_pos   # Right        
            
        if x==self.base_x and y==self.base_y:
            battery=100
            
        self.state = x, y, z, battery
        return self.state

    def _check_limits(self):
        x, y, z, battery = self.state        
        inside = bool(
            x < self.x_min
            or x > self.x_max+2
            or y < self.y_min-2
            or y > self.y_max+2
            or z < self.z_min-2
            or z > self.z_max+2
            or battery<-50
            )  
        return inside

    def _get_map_image(self):
        # make a color map of fixed colors
        #cmap = colors.ListedColormap(['blue', 'green', 'red'])
        #bounds = [0, 1, 3, 5]
        #norm = colors.BoundaryNorm(bounds, cmap.N)
        map_image = "tmp.png"
        our_map = np.array(self.relevance_map)
        our_map[self.base_x, self.base_y] = 100
        our_map = np.rot90(our_map, k=1)
        fig = plt.figure(figsize=(10, 10))
        ax = fig.gca()
        ax.set_xticks(np.arange(-0.5, len(our_map) + 0.5, 1))
        ax.set_yticks(np.arange(-0.5, len(our_map) + 0.5, 1))
        plt.subplots_adjust(0, 0, 1, 1, 0, 0)
        #plt.imshow(our_map, cmap=cmap, norm=norm)

        # our_map += abs(our_map.min())
        # # our_map /= 256
        # our_map = our_map.astype(np.uint8)
        # our_map = cv2.applyColorMap(our_map, cmapy.cmap('viridis'))
        # our_map[self.base_x, self.base_y] = colors.to_rgb('red')
        # plt.imshow(our_map)

        plt.imshow(our_map, interpolation='nearest', cmap=plt.cm.nipy_spectral)
        plt.grid(True)
        ax.axes.xaxis.set_ticklabels([])
        ax.axes.yaxis.set_ticklabels([])
        plt.savefig(map_image)
        plt.close(fig)

        return map_image
    
    def _check_map(self):
        rel_map=self.initial_rm
        if not('numpy.ndarray' in str(type(rel_map))):
            raise TypeError("The relevance map is not defined correctly. It should be numpy.array")
        
        ncol, nrow = rel_map.shape
        
        if (ncol<10) or (ncol<10):
            raise ValueError("The relevance map is too small. The min size is 10x10")
          
    def _check_bases(self):
        rel_map=self.initial_rm
        bs=self.base_stations
        
        if not('numpy.ndarray' in str(type(bs))):
            raise TypeError("The base station map is not defined correctly. It should be numpy.array")
        
        if (bs.shape != rel_map.shape):
            raise ValueError("The base station map should be the relevance map size")    
        
        m,n = self.base_coord.shape
        if m < 1:
            raise ValueError("Minimum one base station should be defined")              
