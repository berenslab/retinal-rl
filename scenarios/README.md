Note: This is a copy of the wiki entry for scenarios, that can be found here: https://github.com/berenslab/retinal-rl/wiki/Scenarios-overview 

# Complexity scenarios

## Apple and Gabor gathering
These are here mostly for puproses of archiving/recreating old results. The naming conventions are also somewhat arbitrary. 
In the apples task `r` refers to the number of red apples initially spawned, `b` refers to blue apples and `g` refers to grass.
`hr100` refers to the reward being defined as current health multiplied by 100. The numbers in the suffices are just internal references due to changing minor things or debugging. 

The assignments to health values are as follows:

NOTE: The exact health contributions might be different, due to some balancing across tasks. So to be sure check the `SCRIPTS` in the `.wad` file! (The ordering of the stimuli/categories is still the same though)

Apple gathering:  
<img width="200" alt="Picture1" src="https://user-images.githubusercontent.com/53050061/155291195-428f2ca7-cab3-4724-9e15-09484f02312f.png">

Gabor gathering:  
<img width="655" alt="assignments_01" src="https://user-images.githubusercontent.com/53050061/155291098-155242f3-79fa-4f5c-a45a-b7af68b26348.png">


## MNIST gathering  

The assignments for MNIST gathering are the following:  
<img width="730" alt="assignments_1" src="https://user-images.githubusercontent.com/53050061/155285361-dc14515b-cf0f-4c49-a9e7-046359091ed0.png">  
Here the particular images given are just example of a category, when spawning stimuli a random image from that category is chosen.

## CIFAR gathering
Here the `01` version has a one-to-one correspondence with MNIST `01` in terms of game mechanics. 

The assignments for CIFAR gathering are the following:
<img width="730" alt="image" src=https://user-images.githubusercontent.com/53050061/155293555-b48c8115-3b2f-4819-afd9-6aa35b114ce6.png>  


# Distractor scenarios

## APPCIFAR gathering
Here the scenarios always come in pairs of `appcifar_apples_gathering_...` and `appcifar_cifar_gathering_...`, where the set of stimuli referenced by the title represents the valent ones. For example in `appcifar_apples_gathering_...` the agent's health is modulated only by apples, collecting CIFAR stimuli has no effect on its health (vice versa for `appcifar_cifar_gathering_...`). Important: The `..._06` scenarios have rewards for all positive and all negative stimuli lumped into two categories. It is also approximately balanced with the other scenarios in terms of cumulative spawned reward, in order to be able to compare the performance better across scenarios.

## APPMNIST gathering
This is the same as above, just using MNIST instead of CIFAR. This might be better, since apples need color separation and MNIST needs shape separation. In the APPCIFAR these are both mixed together.

# Obstacle scenarios
These are all the same as the original MNIST scenario, the only addition is grass obstacles. There are three versions with `100`, `200` and `300` obstacles respectively. 