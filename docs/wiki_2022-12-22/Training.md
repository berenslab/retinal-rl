# Training RL agents

... TODO ...  
See [README](https://github.com/berenslab/retinal-rl/blob/main/README.md) for now.



# Training classification networks

The idea here was to train the same architecture on classification as we did on the RL task. This was used to compare RFs with previous literature (Lindsey et al. 2019) and also to test attribution algorithms (see Analysis page). So basically here the network is initialized and trained from scratch. The architectures are exactly the same if the purely feedforward network is used in the RL task, the only difference being the output layer (6 + 1 for RL/actor+critic and 10 for classification - 1 for each digit). To maintain the same number of parameters in the FC layer, the digits are also padded with a 'doom-like' background, but the actual background shouldn't really matter at least in the MNIST case.

This is all currently implemented in the [`class_encoder_train.ipynb`](https://github.com/berenslab/retinal-rl/blob/main/class_encoder_train.ipynb) notebook. To get optimal performance and nicer-looking receptive fields it is also better to increase the number of epochs (currently 30) to somewhere around 100-500. At the end the notebook saves the trained model in `pytorch/models/name_of_the_model.npy`, which can then be accessed by the analysis scripts.

