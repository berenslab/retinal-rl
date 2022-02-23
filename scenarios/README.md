# Scenarios

## Apple, Gabor and Animal gathering
These are here mostly for puproses of archiving/recreating old results. The naming conventions are also somewhat arbitrary. 
In the apples task `r` refers to the number of red apples initially spawned, `b` refers to blue apples and `g` refers to grass.
`hr100` refers to the reward being defined as current health multiplied by 100. The assignments to health values are as follows:

Apple gathering:  
<img width="200" alt="Picture1" src="https://user-images.githubusercontent.com/53050061/155291195-428f2ca7-cab3-4724-9e15-09484f02312f.png">

Gabor gathering:  
<img width="655" alt="assignments_01" src="https://user-images.githubusercontent.com/53050061/155291098-155242f3-79fa-4f5c-a45a-b7af68b26348.png">

Animal gathering:  
<img width="655" alt="Screenshot 2022-01-24 154715" src="https://user-images.githubusercontent.com/53050061/155291217-3e886afa-16e3-40c2-81b2-c99d201d451a.png">

## MNIST gathering  
For the scenarios with datasets the naming convention is changed, to just be: `dataset_gathering_##` where `##` denotes the version.
For MNIST `01` is the only one widely used, `02` has the same `.wad` file, the only difference is that it implements an explicit `death_penalty` (see `mnist_gathering_02.cfg`).

The assignments for MNIST gathering are the following:  
<img width="730" alt="assignments_1" src="https://user-images.githubusercontent.com/53050061/155285361-dc14515b-cf0f-4c49-a9e7-046359091ed0.png">
Here the particular images given are just example of a category, when spawning stimuli a random image from that category is chosen.

## CIFAR gathering
Here the `01` version has a one-to-one correspondence with MNIST `01` in terms of game mechanics. 
The only difference is in the graphics, using CIFAR and the graphics are also shifted slightly further upward to the center of the visual field.
The `02` version also has exactly the same .wad as `01`, the only difference is that the `living_reward` is set to 0 in the cfg file.
This would originally give a reward to the agent at the end of each episode, which is proportional to the time it managed to survive in that episode. 
Removing this means that the only source of reward is the agent's health. The `03` scenario defines reward in the same way, however it changes the game mechanics.
In this case the two most extreme stimuli (trucks and horses) are spawned as barriers instead of pick-ups. 

The assignments for CIFAR gathering are the following:
<img width="730" alt="image" src=https://user-images.githubusercontent.com/53050061/155293555-b48c8115-3b2f-4819-afd9-6aa35b114ce6.png>
(In the case of `03` the horses and trucks are of course not associated with any contributions to health since they are barriers).

