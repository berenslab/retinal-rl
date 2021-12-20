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
