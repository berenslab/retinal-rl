## Current Installation Procedure

Here are the steps I'm currently going through to get the `retina-rl` project working. First [install anaconda or miniconda](https://docs.anaconda.com/anaconda/install/index.html), and then create the environment
``` bash
conda create retina-rl
conda activate retina-rl
```
I'm using `miniconda`, so some of the following command smight be redundant if you're using `anaconda`.

We use the LTS version of `pytorch` to maximize compatibility
```bash
conda install pytorch torchvision torchaudio cudatoolkit=11.1 -c pytorch-lts -c nvidia
```
and then install `sample-factory` and `vizdoom`
```bash
pip install sample-factory
pip install vizdoom
```
We'll also need some other tools
```bash
conda install -c conda-forge matplotlib
```

Now clone the repo
```bash
https://github.com/berenslab/retina-rl.git
```
