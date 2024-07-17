from typing import Protocol
from retinal_rl.models.neural_circuit import NeuralCircuit
from retinal_rl.models.brain import Brain

class RLEngine(Protocol):
    def initialize():
        ...

    def run_training():
        ...

    def analysis():
        ...


class BrainInterface(Protocol):
    def get_brain() -> Brain:
        ...