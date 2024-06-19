import torch
import torch.nn as nn
import os
from typing import List, Union
from retinal_rl.models.neural_circuit import NeuralCircuit


class Brain(nn.Module):
    """
    This model is intended to be used as the "overarching model" combining several "partial models" - such as encoders, latents, decoders, and task heads - in a specified way.
    """

    def __init__(
        self, circuits: List[Union[NeuralCircuit, dict]], upstream: List[int]
    ) -> None:
        """
        partial_models: list of either objects of type NeuralCircuit or dictionaries defining the individual models
        upstream: list defining the upstream dependencies of each model
        """
        model_configs = []
        self.circuits = []
        self.upstream = upstream
        # assert that upstream should not have integers greater than the index
        # of the current model
        for i, up in enumerate(upstream):
            if up >= i:
                raise ValueError(
                    f"Upstream dependency of model {i} must be an earlier model"
                )

        for m in circuits:
            if isinstance(m, NeuralCircuit):
                model_configs.append(m.config)
                self.circuits.append(m)
            elif isinstance(m, dict):
                model_configs.append(m)
                self.circuits.append(NeuralCircuit.model_from_config(m))
            else:
                raise TypeError(
                    "partial_models can only contain objects of type NeuralCircuit or dict"
                )

        super().__init__()

    def forward(self, x):
        outputs = [None] * len(self.circuits)
        inputs = [None] * len(self.circuits)

        # Initialize the input of the first model
        inputs[0] = x

        for i, model in enumerate(self.circuits):
            if inputs[i] is None:
                raise ValueError(f"No input provided for model {i}")
            outputs[i] = model(inputs[i])
            for j, up in enumerate(self.upstream):
                if up == i:
                    inputs[j] = outputs[i]

        return outputs[-1]

    def save(self, filename, save_cfg=True):
        if not os.path.exists(filename):
            os.mkdir(filename)
        for i, model in enumerate(self.circuits):
            model_path = os.path.join(filename, f"model_{i}")
            model.save(model_path, save_cfg)

    @classmethod
    def load(cls, filename, config_file=None, upstream=None):
        if os.path.isdir(filename):
            partial_models = []
            for i in range(len(os.listdir(filename))):
                model_path = os.path.join(filename, f"model_{i}")
                partial_models.append(NeuralCircuit.load(model_path))
            return cls(partial_models, upstream)
        else:
            return super().load(filename, config_file)

    def is_compatible(self):
        # TODO: Implement compatibility checks (e.g., dimensionality)
        pass

    def scan(self, input_size):
        """
        Runs torchscan on all circuits and concatenates the reports.

        Args:
            input_size (tuple): Size of the input tensor (batch_size, channels, height, width).

        Returns:
            str: The concatenated torchscan reports.
        """
        scan_reports = []
        dummy_input = torch.zeros(input_size)

        for i, model in enumerate(self.circuits):
            report = model.scan(dummy_input.shape)
            scan_reports.append(f"Model {i}: {model.__class__.__name__}\n" + report)
            dummy_input = model(dummy_input)

        return "\n".join(scan_reports)
