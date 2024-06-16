### Imports ###

import argparse


### CLI ###


def get_args():
    parser = argparse.ArgumentParser(
        description="Train a Convolutional Autoencoder with classification"
    )
    parser.add_argument(
        "-f",
        "--num_folds",
        type=int,
        default=5,
        help="Number of folds for cross-validation",
    )
    parser.add_argument(
        "-e", "--num_epochs", type=int, default=10, help="Number of epochs to train"
    )
    parser.add_argument(
        "-w",
        "--recon_weight",
        type=float,
        default=0.5,
        help="Weight to balance reconstruction and classification losses",
    )
    parser.add_argument(
        "-d",
        "--device",
        type=str,
        default="cpu",
        help="Device to use for training (cpu or cuda)",
    )
    parser.add_argument(
        "--train_dir",
        type=str,
        default="train_dir",
        help="Directory containing training results",
    )

    return parser.parse_args()
