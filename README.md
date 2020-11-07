# My first custom drone environment

### Description:
A camera-equipped UAV can fly over the environment to be monitored to optimize the visual coverage of high-relevance areas. 
        
The drone starts ****, and the goal is to adopt a patrolling strategy to opimize the PARTIAL coverage throught time.

### Observation:
     
Let's assume that the map is of grid size WxH. Position of the drone is represented as (grid x index, grid y index), where (0,0) is the top left of the grid ((W-1,H-1) is max value)) z-pos is the hight (Z is maximum flying height);
     
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
6    |  UpLeft
7    |  UpRight 
8    |  DownLeft
9    |  DownRight 

### Reward:

A matrix value   


### Starting State:
We should start on one of the base stations choosen randomly
