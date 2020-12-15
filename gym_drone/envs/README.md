## class DroneEnv
 
Functions

||**Used Functions**|**Used Variables**|**Parameters**|**Changes**|**Returns**|**Description**|
| :-: | :-: | :-: | :-: | :-: | :-: | :-: |
|**get\_map**(*self*,rel\_map)|-|-|rel\_map: NumPy array [m × n]|<p>*self*.relevance\_map </p><p>*self*.initial\_rm </p><p>*self*.x\_max</p><p>*self*.y\_max</p>|**-**|Downloads the relevance map|
|**get\_bases**(*self*,bs)|-|-|<p>bs: </p><p>NumPy array [m × n]</p>|<p>*self.*base\_stations</p><p>*self*.base\_x</p><p>*self*.base\_y</p><p>*self*.base\_coord</p>|*self*.base\_coord|Downloads the base stations coordinates|
|<p>**\_get\_cameraspot()**</p><p></p><p>DEVIDE INTO 2 FUNCTIONS</p>|-|<p>*self*.state</p><p>*self*.x\_max</p><p>*self*.y\_max</p>|-|*self*.state\_matrix|<p>tmp\_cameraspot=</p><p>[p1,p2,p3,p4]</p>|<p>Changes the state in matrix format</p><p></p><p>Returns the coordinates of angles of FOV </p>|
|**reset**(*self*)|*self*.**\_get\_cameraspot**()|<p>*self*.base\_coord</p><p>*self*.initial\_rm</p>|-|<p>*self*.state</p><p>*self*.cameraspot</p><p>*self*.state\_matrix</p><p>*self*.relevance\_map</p>|<p>*self*.state\_matrix</p><p>*self*.cameraspot</p>|Returns the initial state|
|**get\_distance**(*self*, state)|*-*|<p>*self*.relevance\_map</p><p>` `*self*.base\_x</p><p>*self*.base\_y</p>|<p>state = </p><p>[x,y,z,bl]</p><p></p>|-|min\_dist|Calculates the minimum distance to the base station|
|**get\_part\_relmap\_by\_camera** (*self*, camera\_spot)|*-*|*self*.relevance\_map|camera\_spot = [p1,p2,p3,p4]|-|<p>NumPy array</p><p>` `[2z×2z]</p>|Returns the part of relevance map inside the FOV|
|**get\_reward**(*self*, state, new\_state, current\_camera\_spot, new\_cs)|<p>*self.***get\_distance**(\_)</p><p>*self.* **get\_part\_relmap\_by\_camera** (\_)</p><p></p>|*self*.k|<p>old\_state, new\_state = </p><p>[x,y,z,bl]</p><p></p><p>old\_camera\_spot,</p><p>new\_camera\_spot = [p1,p2,p3,p4]</p>|-|reward|Calculates the reward function|
|**zero\_rel\_map**(*self*, camera\_spot)|*-*|*-*|camera\_spot = [p1,p2,p3,p4]|*self*.relevance\_map|-|Changes the covered area of map by zeroes|
|**take\_action**(*self*, action)|*-*|<p>*self*.state</p><p>*self*.delta\_battery</p><p>*self*.delta\_pos</p>|action: int|*self*.state|*self*.state|Changes state by taking action|
|**check\_limits**(*self*)|*-*|<p>*self*.state</p><p>*self*.x\_min</p><p>*self*.x\_max</p><p>*self*.y\_min</p><p>*self*.y\_max</p><p>*self*.z\_min</p><p>*self*.z\_max</p>|-|*-*|inside|Checks if the object is if the agent inside the map limits and it has some battery |
|**step**(*self*, action)|<p>*self*.**take\_action**(*\_*)</p><p>*self*.**check\_limits**()</p><p>*self*.**\_get\_cameraspot**()</p><p>*self*.**\_get\_reward**(\_, \_, \_, \_)</p><p>*self*.**zero\_rel\_map**(\_)</p>|<p>*self*.action\_space</p><p>*self*.state</p><p>*self*.base\_x</p><p>*self*.base\_y</p>|action: int|<p>*self*.state</p><p>*self*.cameraspot</p><p>*self*.state\_matrix</p><p>*self*.observation</p>|<p>observation =*self*.observation</p><p>reward=reward</p><p>done=done</p><p>info={}</p>|Makes one step|
|**get\_map\_image**(*self*)|*-*|<p>*self*.relevance\_map</p><p>*self*.base\_x</p><p>*self*.base\_y</p>|-|*-*|map\_image|Returnes a picture for current map background|
|**render**(*self*, mode='human', show=True)|*self*.**get\_map\_image**()|<p>*self*.state</p><p>*self*.x\_min</p><p>*self*.x\_max</p><p>*self*.viewer</p><p></p>|show: bool|<p>*self*.drone\_trans</p><p>*self*.map\_trans</p><p>*self*.drone\_color</p><p>*self*.axle</p><p>*self*.viewer</p>|*self*.viewer.render(\_)|Visualization|
|**close**(*self*)|*-*|*self*.viewer|-|*self*.viewer|*-*|Closes the window|
|**get\_coverage\_rate**(*self*)|*-*|*self*.relevance\_map|-|*-*|coverage\_rate|Calculates the coverage rate|

