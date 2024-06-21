import torch
import torch.nn as nn
import os
from typing import List, Union, Dict, Tuple
from retinal_rl.models.neural_circuit import NeuralCircuit
import networkx as nx
import yaml


class Brain(nn.Module):
    """
    This model is intended to be used as the "overarching model" (brain)
    combining several "partial models" (circuits) - such as encoders, latents,
    decoders, and task heads - in a specified way.
    """

    def __init__(
        self,
        name: str,
        circuits: Dict[str, Union[NeuralCircuit, Dict]],
        sensors: Dict[str, Tuple[int]],
        connections: List[Tuple[str,str]],
    ) -> None:
        """
        name: The name of the brain.
        circuits: List of NeuralCircuit objects or dictionaries containing the configurations of the models.
        sensors: Dictionary specifying the names and (tensor) shape of the sensors/stimuli.
        connections: List of pairs of strings specifying the graphical structure of the brain.
        """
        super().__init__()

        self._config = locals()
        self._config.pop("circuits")

        self.name = name
        self.circuits : Dict[str, NeuralCircuit] = {}
        self.connectome : nx.DiGraph = nx.DiGraph()
        self.sensors = sensors

        for stim in sensors:
            self.connectome.add_node(stim)

        for crcnm, circuit in circuits.items():
            if isinstance(circuit, NeuralCircuit):
                self.circuits[crcnm] = circuit
            elif isinstance(circuit, dict):
                self.circuits[crcnm] = NeuralCircuit.model_from_config(circuit)
            else:
                raise TypeError(
                    "The circuits parameter should be a list of NeuralCircuit objects or dictionaries."
                )

            self.connectome.add_node(crcnm)

        for connection in connections:
            if connection[0] not in self.connectome.nodes:
                raise ValueError(f"Node {connection[0]} not found in the connectome.")
            if connection[1] not in self.connectome.nodes:
                raise ValueError(f"Node {connection[1]} not found in the connectome.")
            self.connectome.add_edge(connection[0], connection[1])

        # Ensure that the connectome is a directed acyclic graph
        if not nx.is_directed_acyclic_graph(self.connectome):
            raise ValueError("The connectome should be a directed acyclic graph.")

    def forward(self, stimuli: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:

        responses = {}

        for node in nx.topological_sort(self.connectome):
            if node in self.sensors:
                responses[node] = stimuli[node].clone()
            else:
                inputs = []
                for pred in self.connectome.predecessors(node):
                    if pred in self.sensors:
                        inputs.append(stimuli[pred])
                    else:
                        ValueError(f"Input node {pred} to node {node} does not (yet) exist")
                responses[node] = self.circuits[node](*inputs)

        return responses

    def save(self, train_dir):
        pth = os.path.join(train_dir, self.name)
        if not os.path.exists(pth):
            os.mkdir(pth)

        config = self._config
        ymlfl = os.path.join(pth, "config.yaml")
        with open(ymlfl, "w") as f:
            yaml.dump(config, f)

        for name, circuit in self.circuits.items():
            crcpth = os.path.join(pth, name)
            if not os.path.exists(crcpth):
                os.mkdir(crcpth)
            circuit.save(crcpth)

    @classmethod
    def load(cls, brain_dir) -> "Brain":
        ymlfl = os.path.join(brain_dir, "config.yaml")
        with open(ymlfl, "r") as file:
            config = yaml.load(file, Loader=yaml.FullLoader)

        circuits = {}
        if os.path.isdir(brain_dir):
            for circuit_dir in os.listdir(brain_dir):
                circuits[circuit_dir] = NeuralCircuit.load(os.path.join(brain_dir, circuit_dir))
        else:
            # throw error
            ValueError("The provided path is not a directory.")

        return cls(**config, circuits=circuits)

    def is_compatible(self):
        # TODO: Implement compatibility checks (e.g., dimensionality)
        pass

    def scan_circuits(self):
        """
        Runs torchscan on all circuits and concatenates the reports.

        Args:
            input_size (tuple): Size of the input tensor (batch_size, channels, height, width).

        Returns:
            str: The concatenated torchscan reports.
        """
        scan_reports = []
        dummy_stimulus = {}
        for sensor in self.sensors:
            dummy_stimulus[sensor] = torch.rand(self.sensors[sensor])

        dummy_response = self.forward(dummy_stimulus)

        for crcnm, circuit in self.circuits.items():
            report = circuit.scan(dummy_response[crcnm].shape)
            scan_reports.append(f"{crcnm} : {circuit.__class__.__name__}\n\n{report}")

        return "\n".join(scan_reports)
