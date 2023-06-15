Brief tutorial on how to create a custom scenario in ZDoom using Slade and provide links to useful external resources.  
  
# Introduction

Each scenario in ViZDoom is defined by a '.wad' file and a '.cfg' file. Briefly, the '.wad' file is the way ZDoom is run, it defines mods such as custom textures and mechanics. Executing the file while having ZDoom installed launches the modded game. The accompanying '.cfg' file is a much simpler text file that outlines how ViZDoom interacts with the '.wad' file, defining things like i/o, graphics and some reward functionality. This tutorial will focus mainly on creating custom '.wad' files, for '.cfg' see the original ViZDoom documentation ([ViZDoom Configuration Files](https://github.com/mwydmuch/ViZDoom/blob/master/doc/ConfigFile.md)).


Before getting started with the tutorial here's a list of useful tools and resources for creating custom scenarios:

Tools:
- [Slade 3](https://slade.mancubus.net/) (recommended wad editor with better scripting support and availability across platforms unlike for example [DoomBuilder](http://www.doombuilder.com/))
- [ACC Compiler](https://zdoom.org/wiki/ACC) (needs to be set up within SLADE 3 to be able to compile customly written ACS language scripts (to spawn things for example))
- [DoomCrap](http://baghead.drdteam.org/tools.xml) (used for batch image processing when importing large datasets, allows to modify doom-specific 'grAb' chunks in png files (it's an alternative to ['SetPNG'](https://zdoom.org/wiki/SetPNG) or ['grabPNG'](https://forum.zdoom.org/viewtopic.php?f=44&t=19876)))
- [Custom tools](https://github.com/berenslab/retinal-rl/tree/main/scenarios/tools) (used to programmatically reformat datasets and write 'DECORATE.txt' files)

Resources:
- [ZDoom wiki](https://zdoom.org/wiki/Main_Page) (main resource for anything related to modding, most useful for 'DECORATE.txt' and ACS script functionalities)
- [ZDoom forum](https://forum.zdoom.org/index.php) (includes a lot of useful posts from the modding community for implementing troubleshooting new functionalities)
- [ViZDoom documentation](https://github.com/mwydmuch/ViZDoom/tree/master/doc)
- [ViZDoom demo scenarios](https://github.com/mwydmuch/ViZDoom/tree/master/scenarios) (contain a lot of useful example scenarios)
- [retinal-rl scenarios](https://github.com/berenslab/retinal-rl/tree/main/scenarios) (can be used for modified CIFAR and MNIST scenarios, also see the [README](https://github.com/berenslab/retinal-rl/blob/main/scenarios/README.md))
- [YouTube](https://www.youtube.com/) (there are a lot of tutorials, useful for example to get familiar with using wad editor guis)  
<br />

# SLADE 3

As already mentioned above Slade 3 is the recommended wad editor. Additionally, when creating custom environments, it can be useful to import and modify already existing scenarios, for example from [ViZDoom](https://github.com/mwydmuch/ViZDoom/tree/master/scenarios) or [retinal-rl](https://github.com/berenslab/retinal-rl/tree/main/scenarios), since this can avoid a lot of duplication work. This is especially recommended in cases of working with MNIST or CIFAR, since importing these datasets into .wad can be quite laborious (see later sections).

Before starting modding with Slade 3 it's recommended to have a look at the tutorials on their [website](http://slade.mancubus.net/index.php?page=wiki&wikipage=Tutorials) and/or watch a few YouTube tutorials. These are especially useful for setting up ZDoom and Slade, getting familiar with the gui and learning about basic functionalities (creating maps, importing textures etc.). Since there are many resources available online, this will not be considered in this wiki. Instead we will just have a look at the most important parts of a '.wad' file, especially in the context of importing datasets such as MNIST and CIFAR.

The main section within the Slade gui in this context is the 'Entries' section, which contains scripts and graphics:  
<img width="472" alt="image" src="https://user-images.githubusercontent.com/53050061/154514557-c27e60e3-6314-4670-8a50-4b8f1f580f2e.png">  
**Screenshot 1**: SLADE 3 gui for wad editing.

The following sections will explain what is the function of each of these. This tutorial mostly follows the `mnist_gathering_01.wad` as an example, so the references made in the following section can be followed by opening this scenario in SLADE. The wad can be found [here](https://github.com/berenslab/retinal-rl/blob/main/scenarios/mnist_gathering_01.cfg), and it would be recommended to follow this tutorial with the wad file open in Slade.  
Example of what this scenario looks like:
 
![spec_pol](https://user-images.githubusercontent.com/53050061/154523662-0ae70956-e915-42ba-ab48-99cacb331ce1.gif)  
**Gif 1**: Example of the `mnist_gathering_01.wad` scenario.

## MAPINFO
Defines some general features of the mod, in our case these are only `sky1 = 'WIND'` and `music = D_RUNNIN`. The music doesn't matter in this context (agent doesn't get sound input), but the `sky1` variable is important since it points to the `WIND` graphic (see **Screenshot 1**). This swaps the default ZDoom `sky1` graphic with something less hellish (see **gif 1**).

## DECORATE 
This is where most of the magic happens. `DECORATE.txt` is where custom ['actors'](https://zdoom.org/wiki/Category:Actors) are defined. For our purposes these were either 1) positive or negative health pick-ups or 2) obstacles. In future versions this is where one could implement for example moving 'monsters'. In our case all actor definitions are relatively similar and have the following structure:  
```
ACTOR a : Health
{
  +COUNTITEM
  +INVENTORY.ALWAYSPICKUP
  Inventory.Amount -25 //this doesn't work of course ....
  Inventory.MaxAmount 100
  //Inventory.PickupMessage "Picked up a poison!"
  States
  {
  AAAA:
    AAAA A -1
  AAAB:
    AAAB A -1
  AAAC:
    AAAC A -1
```
... (all the different permutations of letters A-Q in last three spots) ...
```
  AQQO:
    AQQO A -1
  AQQP:
    AQQP A -1
  AQQQ:
    AQQQ A -1
  }
}
```
Firstly, the line `ACTOR a : Health` is important, since it determines the name of the actor. In this case the actor name is `a` (this will become important when we come to the `SCRIPTS` part of the wiki. Another thing we can note is that `a` in this case inherits from the class `Health` which is itself a subclass of `Inventory` which is a subclass of `Actor`. When implementing for example monsters, this would of course need to be changed. Secondly, the `States` section defines how `a` will behave in the environment. Under regular conditions, these would be used for example to make a monster run towards the player, start firing its rifle, spawn minions... and each state can be linked to different textures. This is the part that we use to reference the textures in an image database. So if we for example take the first state definition:
```
  AAAB:
    AAAB A -1
```
The first line here corresponds to the state name and the second line corresponds to the graphic and how it will behave. The name of the state corresponds of four letters, where the first one is the capital of the actor name (in this case `a`) and the other three represent a unique combination of alpha(numeric) characters. The actor name refers to a unique class within a dataset and the other three letters are used as a reference for unique entries within an image datasets. Thus, there is a different actor definition for each class `a` to `j` and the particular stimuli are encoded by letters A-Q. To give a concrete example: in MNIST, `a` would correspond to class `0` and `AAA` would be a unique identifier for the first MNIST entry within the class `0`. The name of the class also matches with the first four letters in the second line above. This line is where the actual reference to the texture is made, in this case to `AAABA0` (see **Screenshot 1** below `SS_START`), which corresponds to a .png file (`AAABA0.png`). Here you can also see that `A0` is added to the ending of a 4-letter unique identifier. For our purposes we kept these fixed - the `A` corresponds to the 5th letter in line 2 above and `0` is automatically added by ZDoom when referencing the texture. The last thing about the second line is the `-1`. This entry refers to how long a texture stays on display, and in our case -1 just denotes infinity, since our actor does not change textures during certain time intervals for example. In our case the actor just gets spawned (for example actor `c`, corresponding to digit `2`), the `SCRIPTS.txt` file generates a random string (for example `BGD`) which puts the instantiation of the actor into state `CBGD`. This state is associated with the `CBGDA10.png` texture which is an instantiation of a digit class `2`. Due to `-1` the actor stays in this state, displaying this particular texture, until it is collected by the player.  
  
Another thing to note here is that the letters are in the range of A-Q, which is approximately what is needed to represent all the entries in MNIST and CIFAR. There can be some issues if the unique alphanumerical string has a special property within the `DECORATE` file. For example if the actor is `f` and the string is `AIL` this constitutes `FAIL`, which is an exception that needs to be handled both when making the `DECORATE` file as well as when generating random strings within `SCRIPTS`. Another important thing to note is that multiple `DECORATE.txt` files can be imported into SLADE 3, so normally there would be a unique decorate file for each actor for the sake of convenience when generating these programmatically.

## TEXTURE1 and PNAMES
These contain references to 'non-actor' textures, such as for example `Flat` textures used for floors and ceilings or `Walltexture` used for walls. In our case this is only necessary when importing the green grass texture (see gif above). If such a texture is imported from a .png file, it needs to be transformed to a Doom-compatible format within SLADE (this is not necessary for `actor` textures) and it also needs to be registered within TEXTURE1 and PNAMES. For example see this [tutorial](http://slade.mancubus.net/index.php?page=wiki&wikipage=How-to-Import-Textures).

## MAP01 and TEXTMAP
These are generated/modified automatically when making/changing a map within the SLADE gui using the built in 'Map editor' (green M icon left of the play button in **Screenshot 1**). Generally TEXTMAP just contains a text representation of the map (locations of walls, assignments of textures, special properties of the floor...)

## SCRIPTS and BEHAVIOUR
Besides the decorate file, `SCRIPTS` is where the most important things happen, namely:
1) The `reward` and `shaping_reward` variables are defined and modified, which ViZDoom picks up on to feeds them to Sample-Factory and retinal-rl.
2) Parameters relating to spawning things (limits of the spawning arena, number of initial actors spawned for each category, spawning delay during experience...)
3) Assignments of health contributions to each actor - this is the only source of 'labels' within our setup
4) Scripts that govern the workflow  

Most of the code is written in [Action Code Script (ACS)](https://zdoom.org/wiki/ACS), which is a custom language, but resembles C. This code is generally well documented within the `SCRIPTS` file in for example `mnist_gathering_01.wad`, so for more information on the functionality of different parameters or scripts this would be the best resource. The only thing worth specifically mentioning (as it refers to the 'DECORATE' section above) is the way in which a random state is assigned to an actor upon spawning. This is done within the spawn functions (for example `Spawna()`) where the `GetRandStr()` function is called to generate the unique three-letter string. This is then concatenated with the actor-specific character and the state is set using `SetActivator (a_tid);` and `SetActorState(0, concat);` where `t_id` is a unique identifier for the actor `a` and `concat` is the concatenated string.  
Once modifications are made to `SCRIPTS`, it has to be compiled before the .wad file is run or exported for use within retinal-rl. This can be done by right-clicking on `SCRIPT`, navigating towards `Script` and clicking `Compile ACS`. An ACS compiler of course needs to be set up (see the 'Introduction' section above). The compiled version of `SCRIPTS` is then saved as `BEHAVIOR`.
The usual development procedure was to write modifications to `SCRIPTS`, compile and then run the .wad file within SLADE (play button in the toolbar) and then go back to modifying `SCRIPTS`. Some useful tips for debugging include using `print()` which can display variables on the player's screen during the game (for example `reward` or `concat`). Additionally sometimes it can be useful to turn off constant damage from the floor to for example test the contributions of stimuli to the player's health (this can be done easily by temporarily removing `special=83;` in one of the last lines of `TEXTMAP`.
<br />
<br />
# Importing datasets
The pipeline to import datasets (tried and tested on MNIST and CIFAR) comprises the following steps:
1) Downloading the database, reformatting to .png and reorganising the directory structure
2) Setting 'grAb' chunks for graphic offsets within ZDoom
3) Running `format_ds_slade.ipynb` to rename dataset entries and generate `DECORATE` files
4) Modifying and re-compiling `SCRIPTS`
5) Importing `DECORATE` file and renamed textures
6) Testing

## Preparing dataset
Downloading and reformatting should be quite straightforward. The directory structure that the following tutorial assumes is to have a root folder which contains one subfolder per class within the dataset, each of these containing all entries under that class as .png files. The root folder should also contain `decorate_template.txt` which is used to programmatically generate a `DECORATE` file for each specific actor. `decorate_template.txt` is available in [scenarios/ds_tools](https://github.com/berenslab/retinal-rl/tree/main/scenarios/tools).   
So for mnist the structure should look like this:  

├── mnist_png                    
│   ├── 0          
│   ├── 1  
│   ├── 2          
│   ├── 3  
│   ├── 4          
│   ├── 5  
│   ├── 6          
│   ├── 7  
│   ├── 8           
│   ├── 9         
│   └── decorate_template.txt        

## Setting offsets
When vanilla .png files are imported in ZDoom, their offsets are automatically set to 0:  
<img width="478" alt="image" src="https://user-images.githubusercontent.com/53050061/154540475-3bdf91e6-8309-4c22-9ca6-b26cfe18cee7.png">  
This means that when a thing is spawned, the texture will be displayed on the bottom right with respect to the location of that thing (it will appear as if it is within the ground and shifted right, x axis in image above denotes ground level). In order to correct for this we would want to set the offsets so that the texture is above ground and centred. Ideally we would want our stimuli to look like this:  
<img width="482" alt="image" src="https://user-images.githubusercontent.com/53050061/154541438-742e5c4b-5cc0-4fbd-b12e-fde0fa1d660f.png">

In the case of few stimuli this can be done manually within the SLADE gui (as in the screenshots above), however this becomes impossible for larger datasets. In this case we can use batch image processing tools, in our case from [DoomCrap](http://baghead.drdteam.org/tools.xml). The way such tools work is that they write custom [grAb](https://zdoom.org/wiki/GrAb) chunks to .png files, which specify the offsets (this is exactly what SLADE does on the back-end within the gui). The tool that comes with DoomCrap is essentialy a version of `grabPNG` software, with usage outlined [here](https://forum.zdoom.org/viewtopic.php?t=19876&). For our purposes, we would only need a single command:
```
grabpng -grab *x_offset* *y_offset* path/to/dataset/*.png
```
This will automatically write grAb chunks to all png files within our directory. The offsets should be chosen based on the dataset and are specified in pixels, so in the case of the images above for MNIST (28x28 pixels) we need an x offset of 14 to centre and y offset of 28 to bring the stimulus above the x axis. In some cases however it might be preferable to offset textures further up so that they appear in the centre of the agent's field of view when collecting them - this could improve visual learning.

## Renaming and DECORATE

The next step is to rename our .png files to match the encoding convention that we use to display random stimuli (see DECORATE and SCRIPTS sections above). We will simultaneously also write a `DECORATE` file for each of the stimulus classes. This can be done using `format_ds_slade.ipynb` which can be found under [retinal-rl/scenarios/tools](https://github.com/berenslab/retinal-rl/tree/main/scenarios/tools). The notebook has some comments that describe what it's doing, however certain things might need to be modified before running it. The most important parameter there is `num_uniq`, which determines how many unique identifiers will need to be generated. Currently it is set to 17 which is sufficient for MNIST and CIFAR (17^3 entries per class), but if datasets of different sizes are used this might have to be changed.

## Modifying SCRIPTS

Same as above, in the case of importing a dataset with a different number of entries per class, some changes would have to be implemented in scripts, mainly changing the `alphanum` and `num_uniq` parameters. Additionally if a dataset with number of classes different from 10 is imported, `SCRIPTS` should be modified accordingly (adding or deleting certain parameters and functions/function calls). Additionally other parameters such as the numbers of initially spawned stimuli of each class and the assignments of health contribution to particular classes can be changed.

## Importing to SLADE

When all of the above is completed we are ready to import the `DECORATE` files and the dataset images to SLADE. Currently this is done by dragging and dropping, but it could potentially be improved by [using .zip files instead of .wad](https://zdoom.org/wiki/Using_ZIPs_as_WAD_replacement) (but one would need to check for compatibility with ViZDoom). Firstly the `DECORATE` files should be imported underneath the `MAPINFO` file within the `Entries` section of SLADE (see **Screenshot 1**). Unlike in **Screenshot 1** we would now have 10 `DECORATE` but that is alright as already mentioned before. The textures on the other hand need to be imported between `SS_START` and `SS_END` markers. After `SS_END` there should also be an `ENDMAP` marker.

## Testing

It's important to test if everything worked as intended. Here the debugging tips from the SCRIPTS section above can come in handy. Additionally it's worth checking if the texture offsets have been set correctly.  
<br />

# Running in retinal-rl

Before running in retinal-rl two additional things are required. The first one is a config file accompanying the .wad. Here an important thing to note is that the filenames of the two should match and that the config file should point to the correct .wad file in the `doom_scenario_path` field. The last step is to move both files to the `retinal-rl\scenarios` folder, which then makes them accessible for training agents and running analyses.