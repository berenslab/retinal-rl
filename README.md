# Retinal RL

A deep learning framework for vision research using deep reinforcement learning.

## Apptainer Environment

Retinal-Rl is designed to run in a containerized environment using [Apptainer](https://apptainer.org/docs/user/latest/).

### Installation

1. [Install](https://apptainer.org/docs/admin/main/installation.html) Apptainer to run the containerized environment.

2. Get the container:

- Either pull the pre-built container:
```bash
apptainer pull retinal-rl.sif oras://ghcr.io/berenslab/retinal-rl:singularity-image
```
- or build from source:
```bash
apptainer build retinal-rl.sif resources/retinal-rl.def
```

### Running Experiments

The `scan` command prints info about the proposed neural network architecture:
```bash
apptainer exec retinal-rl.sif python main.py +experiment="{experiment}" command=scan
```
The experiment must always be specified with the `+experiment` flag. To train a
model, use the `train` command:
```bash
apptainer exec retinal-rl.sif python main.py +experiment="{experiment}" command=train
```

**Note:To use the analyze command**

To ensure `command=analyze` works, you need to first specify which run to analyse, only then it can analyse that specific run, else it will fail. By default it will try to analyse the previous run, and if your previous run was not train, which includes analyse function, then it won't work. To enable analysis of a previous file, it can be mentioned in the /retinal-rl/config/user/experiment/****.yaml file. So, whichever experiment yaml file you have, there you can mention the run name to analyse and by default it is set to be run_[current date and time] 

`apptainer` commands can typically be replaced with `singularity` if the latter is rather used.



## Hydra Configuration

The project uses [Hydra](https://hydra.cc/) for configuration management.

### Directory Structure

The structure of the `./config/` directory is as follows:

```
base/config.yaml     # General and system configurations
user/
├── brain/           # Neural network architectures
├── dataset/         # Dataset configurations
├── optimizer/       # Training optimizers
└── experiment/      # Experiment configurations
```

### Default Configuration

Template configs are available under `./resources/config_templates/user/...`, which also provide documentation of the configuration variables themselves. Consult the hydra documentation for more information on [configuring your project](https://hydra.cc/docs/intro/).

### Configuration Management

1. Configuration templates may be copied to the user directory by running:
```bash
bash tests/ci/copy_configs.sh
```

2. Template and custom configurations can be sanity-checked with:
```bash
bash tests/ci/scan_configs.sh
```
which runs the `scan` command for all experiments.

## Weights & Biases Integration

Retinal-RL supports logging to [Weights & Biases](https://wandb.ai/site) for experiment tracking.

### Basic Configuration

By default plots and analyses are saved locally. To enable Weights & Biases logging, add the `logging.use_wandb: True` flag to the command line:
```bash
apptainer exec retinal-rl.sif python main.py +experiment="{experiment}" logging.use_wandb=True command=train
```

### Parameter Sweeps

Wandb [sweeps](https://docs.wandb.ai/guides/sweeps) can be added to `user/sweeps/{sweep}.yaml` and launched from the command line:
```bash
apptainer exec retinal-rl.sif python main.py +experiment="{experiment}" +sweep="{sweep}" command=sweep
```

Typically the only command line arguments that need a `+` prefix will be `+experiment` and `+sweep`. Also note that `.yaml` extensions are dropped at the command line.

