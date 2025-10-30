import torch
from captum.attr import InputXGradient

from retinal_rl.models.brain import Brain


def l1_attribution(
    brain: Brain,
    stimuli: dict[str, torch.Tensor],
    target_circuit: torch.Tensor,
    target_output_index: int = 0,
) -> dict[str, torch.Tensor]:
    input_grads: dict[str, torch.Tensor] = {}
    output = brain(stimuli)[target_circuit][target_output_index]
    loss = torch.nn.L1Loss()(output, torch.zeros_like(output))
    loss.backward()
    for key, value in stimuli.items():
        input_grads[key] = value.grad.detach().cpu()
    return input_grads


def captum_attribution(
    brain: Brain,
    stimuli: dict[str, torch.Tensor],
    target_circuit: torch.Tensor,
    target_output_index: int = 0,
) -> dict[str, torch.Tensor]:
    input_grads: dict[str, torch.Tensor] = {}

    stimuli_keys = list(stimuli.keys()) # create list to preserve order
    print(stimuli_keys)
    def _forward(*args: tuple[torch.Tensor]) -> torch.Tensor:
        print(len(args))
        print([type(a) for a in args])
        assert len(args) == len(stimuli_keys)
        return brain({k: v for k, v in zip(stimuli_keys, args)})[
            target_circuit
        ][target_output_index]

    value_grad_calculator = InputXGradient(_forward)
    value_grads = value_grad_calculator.attribute(tuple(stimuli[k] for k in stimuli_keys))
    for key, value_grad in zip(stimuli_keys, value_grads):
        input_grads[key] = value_grad.detach().cpu()
    return input_grads

def _rescale_zero_one(x: torch.Tensor) -> torch.Tensor:
    _min = torch.min(x)
    _max = torch.max(x)
    return (x - _min) / (_max - _min)

def analyze(
    brain: Brain,
    stimuli: dict[str, torch.Tensor],
    target_circuit: torch.Tensor,
    target_output_index: int = 0,
    method: str = "l1",
    sum_channels: bool = True,
    rescale_per_frame: bool = False,
) -> dict[str, torch.Tensor]:
    is_training = brain.training

    grad_enabled = torch.is_grad_enabled()
    if not grad_enabled:
        torch.set_grad_enabled(True)
    if not is_training:
        brain.train()
        brain.requires_grad_(False)

    for key, value in stimuli.items():
        stimuli[key] = value.requires_grad_(True)

    input_grads: dict[str, torch.Tensor] = {}
    if method == "l1":
        input_grads = l1_attribution(
            brain, stimuli, target_circuit, target_output_index
        )
    elif method == "attribution":
        input_grads = captum_attribution(
            brain, stimuli, target_circuit, target_output_index
        )

    if sum_channels:
        for key, grad in input_grads.items():
            input_grads[key] = grad.sum(dim=1, keepdim=True)
    if rescale_per_frame:
        for key, grad in input_grads.items():
            for frame in range(grad.shape[0]):
                input_grads[key][frame] = _rescale_zero_one(input_grads[key][frame])

    if not is_training:
        brain.requires_grad_(True)
        brain.eval()
    if not grad_enabled:
        torch.set_grad_enabled(False)
    return input_grads


def plot():  # -> Figure:
    # TODO: Implement plotting logic
    pass
