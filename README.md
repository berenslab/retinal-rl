## Current Installation Procedure

Here are the steps I'm currently going through to get the `retina-rl` project working. First [install anaconda or miniconda](https://docs.anaconda.com/anaconda/install/index.html), and then create the environment
``` bash
conda create retina-rl
conda activate retina-rl
```

We use the LTS version of `pytorch` to maximize compatibility
```bash
conda install pytorch torchvision torchaudio cudatoolkit=11.1 -c pytorch-lts -c nvidia
```
and then install `sample-factory` and `vizdoom`
```bash
pip install sample-factory
pip install vizdoom
```

Now clone the repo
```bash
https://github.com/berenslab/retina-rl.git
```
