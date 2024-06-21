import os
from abc import ABC

import torch
import torch.nn as nn
import yaml
import importlib
import torchscan


class NeuralCircuit(nn.Module, ABC):
    def __init__(self, init_params: dict = {}) -> None:
        """
        Initializes the base model.
        All params in the dictionary will be added as instance parameters.

        init_params: the parameters used to instantiate a model. Simplest way to pass them on: call locals()
        """
        super().__init__()

        # Store all parameters as config
        self._config = init_params
        if "self" in self._config:
            self._config.pop("self")
        if "__class__" in self._config:
            self._config.pop("__class__")
        self.__dict__.update(self._config)

    @property
    def config(self) -> dict:
        conf = self._config
        return {
            "type": self.__class__.__name__,
            "module": self.__class__.__module__,
            "config": conf,
        }

    def save(self, circuit_dir):
        config = self.config
        ymlfl = os.path.join(circuit_dir, "config.yaml")
        whgtfl = os.path.join(circuit_dir, "weights.pth")
        with open(ymlfl, "w") as f:
            yaml.dump(config, f)
        torch.save(self.state_dict(), whgtfl)

    def scan(self, input_size):
        """
        Runs torchscan on the model.

        Args:
            input_size (tuple): Size of the input tensor (batch_size, channels, height, width).

        Returns:
            str: The torchscan report.
        """
        return torchscan.summary(self, input_size, receptive_field=True)

    @staticmethod
    def model_from_config(config: dict):
        _module = importlib.import_module(config["module"])
        _class = getattr(_module, config["type"])
        return _class(**config["config"])

    @staticmethod
    def load(circuit_dir):
        ymlfl = os.path.join(circuit_dir, "config.yaml")
        wghtfl = os.path.join(circuit_dir, "weights.pth")
        with open(ymlfl, "r") as file:
            config = yaml.load(file, Loader=yaml.FullLoader)
        model = NeuralCircuit.model_from_config(config)

        if os.path.exists(wghtfl):
            try:
                model.load_state_dict(torch.load(wghtfl))
            except:
                model.load_state_dict(
                    torch.load(wghtfl, map_location=torch.device("cpu"))
                )
        return model

    @staticmethod
    def str_to_activation(act: str) -> nn.Module:
        act = str.lower(act)
        if act == "elu":
            return nn.ELU(inplace=True)
        elif act == "relu":
            return nn.ReLU(inplace=True)
        elif act == "tanh":
            return nn.Tanh()
        elif act == "softplus":
            return nn.Softplus()
        elif act == "identity":
            return nn.Identity(inplace=True)
        else:
            raise Exception("Unknown activation function")

    @staticmethod
    def calc_num_elements(module: nn.Module, module_input_shape: tuple[int]):
        shape_with_batch_dim = (1,) + module_input_shape
        some_input = torch.rand(shape_with_batch_dim)
        num_elements = module(some_input).numel()
        return num_elements
