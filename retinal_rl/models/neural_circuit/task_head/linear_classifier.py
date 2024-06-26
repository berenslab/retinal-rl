import torch
from retinal_rl.models.neural_circuit import NeuralCircuit

class LinearClassifier(NeuralCircuit):
    def __init__(
        self,
        inp_size: int,
        out_size: int,
        act_name: str,
    ):
        super().__init__()

        self.inp_size = inp_size
        self.out_size = out_size
        self.act_name = act_name
        self.fc = torch.nn.Linear(inp_size, out_size)

    def forward(self, x):
        x = self.fc(x)
        if not self.training:
            x = torch.nn.functional.softmax(x, dim=1)
        return x
