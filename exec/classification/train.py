### Imports ###

import torch

from torchvision.datasets import CIFAR10
from torchvision.transforms import transforms

from retinal_classification.training import cross_validate, save_results
from retinal_classification.arguments import get_args


### Main ###


def main():

    # Get arguments
    args = get_args()

    # Load CIFAR-10 dataset
    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
    )
    dataset = CIFAR10(root="./cache", train=True, download=True, transform=transform)

    # Run cross-validation
    models, histories = cross_validate(
        torch.device(args.device),
        args.num_folds,
        args.num_epochs,
        args.recon_weight,
        dataset,
    )

    # Save results
    save_results(models, histories, args.train_dir, args.recon_weight)


if __name__ == "__main__":
    main()
