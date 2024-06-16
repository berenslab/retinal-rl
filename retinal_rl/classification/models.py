### Imports ###

from typing import Protocol, Tuple
import torch

import torch.nn as nn


### Model definition ###


class AutoencodingClassifier(Protocol):
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]: ...

    def to(self, device: torch.device) -> "AutoencodingClassifier": ...

    def train(
        self: "AutoencodingClassifier", mode: bool = True
    ) -> "AutoencodingClassifier": ...

    def eval(self: "AutoencodingClassifier") -> "AutoencodingClassifier": ...

    def parameters(self) -> torch.nn.parameter.Parameter: ...

    def __call__(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]: ...


class ConvAutoencoder(nn.Module):
    def __init__(self):
        super(ConvAutoencoder, self).__init__()
        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 16, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 32, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),
            nn.ReLU(),
        )
        # Decoder
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(64, 32, 3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 16, 3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(16, 3, 3, stride=2, padding=1, output_padding=1),
            nn.Sigmoid(),  # Assuming input images are scaled between 0 and 1
        )
        # Classifier
        self.classifier = nn.Linear(64 * 4 * 4, 10)  # Adjust the size accordingly

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        encoded_flat = encoded.view(encoded.size(0), -1)
        classification = self.classifier(encoded_flat)
        return classification, decoded
