# My first custom drone environment

### Description:
A camera-equipped UAV can fly over the environment to be monitored to optimize the visual coverage of high-relevance areas. 
        
The drone starts ****, and the goal is to adopt a patrolling strategy to opimize the PARTIAL coverage throught time.

### Observation:
     
Let's assume that the map is of grid size WxH (5x5). Position of the drone is represented as (grid x index, grid y index), where (0,0) is the top left of the grid ((W-1,H-1)(4,4) is max value)) z-pos is the hight (Z is maximum flying height);
     
Type: Box(4)   
Num     Observation               Min                     Max
0       Current x-pos              0                    (W-1)=4
1       Current y-pos              0                    (H-1)=4
2       Current z-pos              0                    (Z-1)=4
3       Terrain angle              0                      2*pi 

Actions:
Type: Discrete(9)
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

Reward:
A matrix value   

Starting State:
All observations are assigned a uniform random value in [1..2]

### Video Box
![alt text](https://github.com/AlinaKasiuk/my_drone_env_1/blob/main/drone_1.png width="350")
