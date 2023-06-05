spreadsheet log for all experiments: https://bit.ly/3E42X9R

12.12.2021:
- `simple` and `lindsey` networks can generally solve the battle environment.
- Performance is generally CPU bottle-necked for `simple` networks, but become GPU bottle-necked for `lindsey` when `vvs_depth > 0`.

12.16.2021:
- First batch of maps with format `apple_gathering_rx_by_gz.wad` proved difficult to train. Reward functions in these maps were based entirely on time alive, which in some sense is an all or nothing signal at time of death.
- Adding reward for current health level appears to address this.

12.17.2021:
- Using the new value function and the `simple` encoder can solve the harder environments (r=30, b=2, g=100) in relatively few iterations (~0.2G)
- Reproduced on cin servers - `lindsey` can quite easily solve the easy red apples task (~0.03G)

12.19.2021:
- running some `lindsey` simulations on both scenarios, observations during training: the agents that walk backwards show two oscillating strategies during learnin, walking backwards away from the arena (low reward periods) or backwards circulating around arena (higher reward periods)

12.20.2021:
- `lindsey03_hr100_r30_b2_g100` ran through, needed 2d 10h and did not converge to a good solution, `lindsey05_hr100_r30_b2_g100` converged to good solution very fast (daster than `simple`)
- added scenarios as `_nb` ('no backwards') at the end of name, might be worth trying

12.26.2021:
- `lindsey05_hr100_r30_b2_g100` ran through (10G) and seems to have developed some interesting receptive fields (red/blue detectors - filter 0, and maybe some edge/horizon detectors - filters 8 and 15?, see `receptive-fields-2021-12-26T12:57:53.png`) - nothing similar can be seen in `lindsey03...` which had the same parameters, but did not solve the task!
-  `lindsey` networks take >2d on gpu25 (~5d on gpu10) to run 10G steps
- tried training `lindsey10` on an `_nb` scenario for 0.3G steps but the performance seemed even worse than in scenarios with all movement options

01.02.2022:
- `lindsey08` crashed in a strange way where samples were being collected but performance stayed constant (maybe related to crash that Kerol reported, since its the same gpu)
- still no luck training `lindsey` models with `vvs=1` after 6 attempts for 0.3G steps

01.07.2022:
- `lindsey` models with `vvs=1` and smaller values for `global_channels` (and `retinal_bottleneck`) manage to solve the task (3/3 for now), the model with 2 `global_channels` learned faster than the model with 1, they also have somewhat interpretable receptive fields even when they are not fully trained

01.09.2022:
- A `lindsey` model with `vvs=1` and 1 channel can solve `health_gathering_supreme` with minimal training (2 runs tried). May need to think about how to make the task more complex.

01.10.2022:
- `lindsey` with `vvs=0` on `health_gathering_supreme` seems unable to learn the problem (> 1G samples).
- Interestingly, `lindsey` with `vvs_depth=0` and 1 channel make good progress on both `battle` and `battle2` (> .5G).

01.11.2022:
- `lindsey` with `vvs=0` and default channels can solve health gathering supreme.

01.19.2022:
- new `gabor_gathering` task is harder but managable, agents take slightly longer to learn
- set up `retinal_rl` with updated environment on gpu10 and gpu25, works well. Also changed naming conventions for docker crates, conda environment etc.
- implemented `linear` encoder that feeds pixels straight to the fc layer/rnn (size determined by `hidden_size`)
- running a set of shorter 0.5G simulations testing the requirement for `hidden_size` in the `gabor_gathering` task (8, 64 and default 512)

01.21.2022:
- `animals` task seems quite challenging. Training with a `vvs_depth=0` produced modest task learning, and interesting receptive fields. Receptive fields appeared to capture specific edge structures of animals. Tried training on `vvs_depth=1` but got stuck in the "walking backwards" local minimum.

01.26.2022:
- some networks managed to solve the `animal_gathering_3` taks (`3` has some modifications in terms of health values, fewer stimuli and bigger textures compared to the original) rfs seem interesting, but still the simplest architectures can manage
- implemented some tensorboard compatibility - graph of encoder and saving rf images
- modified encoder implementation to improve graph + have it defined more 'conventionally'
- tagged changes with v1.0 (some incompatibility with previous version due to encoder changes)

01.28.2022:
- also added `mosaic` network from now on also:
    - changed naming convention, dropping details about reward and task (convention: `network##`)
    - using `reward_scale = 0.1`
    - using shaping reward
    - can use higher resolution (`144x256`)

02.02.2022:
- added demo `mnist_gathering` scenario