import torch
import torch.nn as nn
from typing import List, Dict
from retinal_rl.models.neural_circuit import NeuralCircuit
import networkx as nx
from dataclasses import dataclass


@dataclass
class BrainConfig:
    name: str
    circuits: Dict[str, NeuralCircuit]
    sensors: Dict[str, List[int]]
    connections: List[List[str]]

class Brain(nn.Module):
    """
    This model is intended to be used as the "overarching model" (brain)
    combining several "partial models" (circuits) - such as encoders, latents,
    decoders, and task heads - in a specified way.
    """

    def __init__( self, cfg: BrainConfig) -> None:
        """
        name: The name of the brain.
        circuits: List of NeuralCircuit objects or dictionaries containing the configurations of the models.
        sensors: Dictionary specifying the names and (tensor) shape of the sensors/stimuli.
        connections: List of pairs of strings specifying the graphical structure of the brain.
        """
        super().__init__()

        self.name = cfg.name
        self.circuits = nn.ModuleDict(cfg.circuits)
        self.connectome : nx.DiGraph = nx.DiGraph()
        self.sensors = {}
        for sensor, shape in cfg.sensors.items():
            self.sensors[sensor] = tuple(shape)

        for stim in self.sensors:
            self.connectome.add_node(stim)

        for crcnm in self.circuits:
            self.connectome.add_node(crcnm)

        for connection in cfg.connections:
            if connection[0] not in self.connectome.nodes:
                raise ValueError(f"Node {connection[0]} not found in the connectome.")
            if connection[1] not in self.connectome.nodes:
                raise ValueError(f"Node {connection[1]} not found in the connectome.")
            self.connectome.add_edge(connection[0], connection[1])

        # Ensure that the connectome is a directed acyclic graph
        if not nx.is_directed_acyclic_graph(self.connectome):
            raise ValueError("The connectome should be a directed acyclic graph.")

    def calculate_inputs(self, node : str, responses: Dict[str, torch.Tensor]) -> torch.Tensor:

        inputs = []
        input = torch.Tensor()
        for pred in self.connectome.predecessors(node):
            if pred in responses:
                inputs.append(responses[pred])
            else:
                ValueError(f"Input node {pred} to node {node} does not (yet) exist")
        if len(inputs) == 0:
            ValueError(f"No inputs to node {node}")
        elif len(inputs) == 1:
            input = inputs[0]
        else:
            input = torch.cat(inputs, dim=1)
        return input

    def forward(self, stimuli: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:

        responses = {}

        for node in nx.topological_sort(self.connectome):
            if node in self.sensors:
                responses[node] = stimuli[node].clone()
            else:
                input = self.calculate_inputs(node, responses)
                responses[node] = self.circuits[node](input)
        return responses

    # def save(self, train_dir):
    #     pth = os.path.join(train_dir, self.name)
    #     if not os.path.exists(pth):
    #         os.mkdir(pth)
    #
    #     config = self._config
    #     ymlfl = os.path.join(pth, "config.yaml")
    #     with open(ymlfl, "w") as f:
    #         yaml.dump(config, f)
    #
    #     for name, circuit in self.circuits.items():
    #         crcpth = os.path.join(pth, name)
    #         if not os.path.exists(crcpth):
    #             os.mkdir(crcpth)
    #         circuit.save(crcpth)

    # @classmethod
    # def load(cls, brain_dir) -> "Brain":
    #     ymlfl = os.path.join(brain_dir, "config.yaml")
    #     with open(ymlfl, "r") as file:
    #         config = yaml.load(file, Loader=yaml.FullLoader)
    #
    #     circuits = {}
    #     if os.path.isdir(brain_dir):
    #         for circuit_dir in os.listdir(brain_dir):
    #             circuits[circuit_dir] = NeuralCircuit.load(os.path.join(brain_dir, circuit_dir))
    #     else:
    #         # throw error
    #         ValueError("The provided path is not a directory.")
    #
    #     return cls(**config, circuits=circuits)

    def is_compatible(self):
        # TODO: Implement compatibility checks (e.g., dimensionality)
        pass

    def scan_circuits(self):
        """
        Runs torchscan on all circuits and concatenates the reports.
        """
        # Print connectome
        print("\n\nConnectome:\n")
        print("Nodes: ", self.connectome.nodes)
        print("Edges: ", self.connectome.edges)

        # Run scans on all circuits
        dummy_stimulus = {}
        for sensor in self.sensors:
            dummy_stimulus[sensor] = torch.rand(self.sensors[sensor])

        dummy_response = self.forward(dummy_stimulus)

        for crcnm, circuit in self.circuits.items():
            print(f"\n\nCircuit Name: {crcnm}, Class: {circuit.__class__.__name__}\n")
            dummy_input = self.calculate_inputs(crcnm, dummy_response)
            circuit.scan(dummy_input.shape)
