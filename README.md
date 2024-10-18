# Retinal RL

## Setting up the development environment

We provide a singularity / apptainer container which should always be up to date and allow to run the code immediately. You do not need to build it yourself (but can, of course), you can just pull it!

### Install Singularity / Apptainer

First you need to install [apptainer](https://github.com/apptainer/apptainer/) (or singularity) in order to run code.

### Get the container

Once you have apptainer installed, you can simply pull the container

```bash
apptainer pull retinal-rl.sif oras://ghcr.io/berenslab/retinal-rl:singularity-image
```

or try to build it on your own (no advantages of doing that, except you want to change some dependency in the .def file):

```bash
apptainer build retinal-rl.sif resources/retinal-rl.def
```

### Prepare config directory for experiments

The repository comes with some example configuration files, which you find under 'resources/config_templates'. For running experiments however, they need to be in 'config'.
You can either copy them there by hand or run the following script from the top-level directory:

```bash
bash tests/ci/copy_configs.sh
```

### Test basic functionality

Now you are basically ready to run experiments!
To test that everything is working fine, you can run:

```bash
bash tests/ci/scan_configs.sh
```

The script loops over all experiments defined in config/experiment and runs a "scan" on them.
If instead you want to run a single experiment file, run:

```bash
apptainer exec retinal-rl.sif python main.py +experiment="$experiment" command=scan system.device=cpu
```

## Running retinal RL simulations [DEPRECATED]

There are three main scripts for working with `retinal-rl`:

- `train.py`: Train a model.
- `analyze.py`: Generate some analyses.
- `enjoy.py`: Watch a real time simulation of a trained agent.

Each script can be run by python in `python -m {script}`, where {script} is the name of the desired script (without the `.py` extension), followed by a number of arguments. Note that `train.py` must always be run first to create the necessary files and folders, and once run will permanently set most (all?) of the arguments of the simulation, and will ignore changes to these arguments if training is resumed.

Certain arguments must always be provided, regardless of script, namely:

- `--env`: Specifies the desired map. This will always have the form `retinal_{scenario}`, where scenario is the shared name of one of the `.wad`/`.cfg` file pairs in the `scenarios` directory.
- `--algo`: The training algorithm; for now this should always be `APPO`.
- `--experiment`: The directory under the `train_dir` directory where simulation results are saved.

The following argument should always be set when training for the first time:

- `--encoder_custom`: The options are `simple`, which is a small, hard-coded network that still tends to perform well, and `lindsey`, which has a number of tuneable hyperparameters.

For specifying the form of the `lindsey` network, the key arguments are:

- `--global_channels`: The number of channels in each CNN layers, except for the bottleneck layer.
- `--retinal_bottleneck`: Number of channels in the retinal bottleneck.
- `--vvs_depth`: Number of CNN layers in the ventral stream network.
- `--kernel_size`: Size of the kernels.

Finally, when training a model there are a number of additional parameters for controlling the reinforcement learning brain, and adjusting simulation parameters. The key ones to worry about are

- `--hidden_size`: The size of the hidden/latent state used to represent the RL problem.
- `--num_workers`: This is the number of simulation threads to run. This shouldn't be more than the number of cores on the CPU, and can be less if the simulation is GPU bottlenecked.
