import torch
from captum.attr import InputXGradient

from retinal_rl.models.brain import Brain
from retinal_rl.util import rescale_zero_one


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

    stimuli_keys = list(stimuli.keys())  # create list to preserve order

    def _forward(*args: tuple[torch.Tensor]) -> torch.Tensor:
        assert len(args) == len(stimuli_keys)
        return brain({k: v for k, v in zip(stimuli_keys, args)})[target_circuit][
            target_output_index
        ]

    value_grad_calculator = InputXGradient(_forward)
    value_grads = value_grad_calculator.attribute(
        tuple(stimuli[k] for k in stimuli_keys)
    )
    for key, value_grad in zip(stimuli_keys, value_grads):
        input_grads[key] = value_grad.detach().cpu()
    return input_grads


ATTRIBUTION_METHODS = {"l1": l1_attribution, "attribution": captum_attribution}


def analyze(
    brain: Brain,
    stimuli: dict[str, torch.Tensor],
    target_circuit: torch.Tensor,
    target_output_index: int = 0,
    method: str = "l1",
    sum_channels: bool = True,
    rescale_per_frame: bool = False,
) -> dict[str, torch.Tensor]:
    assert method in ATTRIBUTION_METHODS, f"Unknown attribution method: {method}"

    is_training = brain.training
    required_grad = next(brain.parameters()).requires_grad
    grad_enabled = torch.is_grad_enabled()

    # this is required to compute gradients
    torch.set_grad_enabled(True)
    brain.train()
    brain.requires_grad_(False)

    for key, value in stimuli.items():
        stimuli[key] = value.requires_grad_(True)

    input_grads: dict[str, torch.Tensor] = {}
    input_grads = ATTRIBUTION_METHODS[method](
        brain, stimuli, target_circuit, target_output_index
    )

    if sum_channels:
        for key, grad in input_grads.items():
            input_grads[key] = grad.sum(dim=1, keepdim=True)
    if rescale_per_frame:
        for key, grad in input_grads.items():
            for frame in range(grad.shape[0]):
                input_grads[key][frame] = rescale_zero_one(input_grads[key][frame])

    # restore original state of training / grad_enabled
    brain.requires_grad_(required_grad)
    brain.train(is_training)
    torch.set_grad_enabled(grad_enabled)
    return input_grads


def plot():  # -> Figure:
    # TODO: Implement plotting logic
    raise NotImplementedError
