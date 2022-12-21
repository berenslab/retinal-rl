import matplotlib.pyplot as plt
import numpy as np
import matplotlib.transforms as mtransforms

### Code to generate side view schematic of a map (for figure 1)

# parameters
n_r = 21 # number of stimuli to spawn (in all tasks this is 21 for positive)
n_b = 9 # same, but 9 for negative
d = 10 # length of square side of spawned stimulus on the map surface
l = 1000 # length of square arena side

floor_rgb = (170/256, 225/256, 170/256)
red_rgb = (1, 0, 0)
blue_rgb = (0, 0, 1)

# initialising map
map = np.ones((l, l, 3))

# setting green floor
for (i, c) in enumerate(floor_rgb):
    map[:,:,i] = c

# setting red apples
for i in range(n_r):
    x, y = np.random.randint(0,l,size=2)
    map[x-d:x+d,y-d:y+d,:] = red_rgb

# setting blue apples
for i in range(n_b):
    x, y = np.random.randint(0,l,size=2)
    map[x-d:x+d,y-d:y+d,:] = blue_rgb
    
fig, ax = plt.subplots(1, dpi=300)
ax.imshow(map, transform=mtransforms.Affine2D().skew_deg(30, 30).rotate_deg(-45) + ax.transData)
plt.axis('off')
plt.xlim((-100,2300))
plt.ylim((-500,500))
plt.show()