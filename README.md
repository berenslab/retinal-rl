## Current Installation Procedure

Here are the steps I'm currently going through to get the `retinal-rl` project working. First [install anaconda or miniconda](https://docs.anaconda.com/anaconda/install/index.html), and then create the environment
``` bash
conda create --name retinal-rl python=3.8 pip
conda activate retinal-rl
```
I'm using `miniconda`, so some of the following commands might be redundant if you're using `anaconda`.

Now, we use the LTS version of `pytorch` to maximize compatibility
```bash
conda install pytorch torchvision torchaudio cudatoolkit=11.1 -c pytorch-lts -c nvidia
```
and then install `sample-factory` and `vizdoom`
```bash
pip install sample-factory
pip install vizdoom
```
Note, you may require `sample-factory=1.121.4` on a server. We'll also need some other tools and libraries
```bash
conda install -c conda-forge matplotlib pyglet imageio
pip install pygifsicle
```
Finally, if you're missing some fundamental libraries, the following may help:
```bash
conda install -c conda-forge gxx boost
```

Now clone the repo
```bash
https://github.com/berenslab/retinal-rl.git
```
There are three main scripts for working with `retinal-rl`:

- `train.py`: Train a model.
- `analyze.py`: Generate some analyses; right now saves a simulation as a .gif file and renders receptive fields at the output of the first convolutional layer.
- `enjoy.py`: Watch a real time simulation of a trained agent.

Each script can be run by python in `python -m {script}`, where {script} is the name of the desired script (without the `.py` extension), followed by a number of arguments. Note that `train.py` must always be run first, and once run will permanently set most (all?) of the arguments of the simulation, and ignore changes to these arguments if training is resumed.

Certain arguments must always be provided, regardless of script, namely:

- `--env`: Specifies the desired map. Right now this will always have the form `doom_{scenario}`, where scenario is the shared name of one of the `.wad`/`.cfg` file pairs in the `scenarios` directory.
- `--algo`: The training algorithm; for now this should always be `APPO`.
- `--experiment`: The directory where simulation results are saved, which itself lives in the `train_dir` directory.

The following argument should always be set when training for the first time, as it specifies that we're using our encoding model based on Lindsey et. al. 2019:

- `--encoder_custom`: The options are `simple`, which is a small, hard-coded network that still tends to perform well, and `lindsey`, is has a number of tuneable hyperparameters.

For specifing the form of the `lindsey`, the key arguments are:

- `--global_channels`: The number of channels in each CNN layers, except for the bottleneck layer.
- `--retinal_bottleneck`: Number of channels in the retinal bottleneck.
- `--vvs_depth`: Number of CNN layers in the ventral stream network.
- `--kernel_size`: Number of CNN layers in the ventral stream network.

Finally, when training a model there are a number of additional parameters for controlling the reinforcement learning brain, and adjusting simulation parameters. The key ones to worry about are

- `--hidden_size`: The size of the hidden/latent state used to represent the RL problem.
- `--num_workers`: This is the number of simulation threads to run, and should match the number of cores on the CPU.
- `--num_envs_per_worker`: This is the number of environments to simulate per thread. This should be adjusted. `24` seems good.
- `--batch_size`: Also manages per worker load. Try `4096`.
