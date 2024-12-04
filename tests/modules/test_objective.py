import numpy as np
import pytest
import torch
from hydra.utils import instantiate
from omegaconf import DictConfig

from retinal_rl.classification.loss import ClassificationContext
from retinal_rl.models.brain import Brain
from retinal_rl.models.objective import Objective
from runner.util import create_brain


@pytest.mark.skip(reason="Just a helper function")
def run_classification_objective(
    brain: Brain, objective: Objective[ClassificationContext]
):
    # create some random tensors to have sth for in / output
    input = torch.zeros(1, *brain.sensors["vision"])
    label = torch.ones(1, 10) / 10

    # forward pass
    stimuli = {"vision": input}
    responses = brain(stimuli)

    context = ClassificationContext(
        sources=input, inputs=input, classes=label, responses=responses, epoch=1
    )
    return objective.backward(context)


@pytest.mark.skip(reason="Just a helper function")
def grad_sum(model: torch.nn.Module):
    return np.sum(
        [
            0 if param.grad is None else np.sum(param.grad.numpy())
            for param in model.parameters()
        ]
    )


def test_classification_objective(classification_config: DictConfig):
    brain = create_brain(classification_config.brain)
    objective: Objective[ClassificationContext] = instantiate(
        classification_config.optimizer.objective, brain=brain
    )
    assert isinstance(objective, Objective)

    assert grad_sum(brain) == 0

    loss_dict = run_classification_objective(brain, objective)

    assert loss_dict["percent_correct"] == 0 or loss_dict["percent_correct"] == 1
    assert (
        grad_sum(brain) != 0
    ), "Objective should change the gradients, but it's still 0."


def test_objective_all():
    brain_conf = DictConfig(
        {
            "name": "feedforward",
            "sensors": {
                "vision": [
                    3,
                    120,
                    160,
                ]
            },
            "connections": [["vision", "encoder"], ["encoder", "classifier"]],
            "circuits": {
                "encoder": {
                    "_target_": "retinal_rl.models.circuits.convolutional.ConvolutionalEncoder",
                    "num_layers": 3,
                    "num_channels": [4, 8, 16],
                    "kernel_size": 6,
                    "stride": 2,
                    "activation": "relu",
                },
                "classifier": {
                    "_target_": "retinal_rl.models.circuits.fully_connected.FullyConnected",
                    "output_shape": [10],
                    "hidden_units": [128],
                    "activation": "relu",
                },
            },
        }
    )

    obj_conf = DictConfig(
        {
            "_target_": "retinal_rl.models.objective.Objective",
            "losses": [
                {
                    "_target_": "retinal_rl.classification.loss.ClassificationLoss",
                    "target_circuits": ["__all__"],
                }
            ],
        }
    )

    brain = create_brain(brain_conf)

    objective: Objective[ClassificationContext] = instantiate(obj_conf, brain=brain)

    assert grad_sum(brain) == 0
    run_classification_objective(brain, objective)
    assert (
        grad_sum(brain) != 0
    ), "Objective should change the gradients, but it's still 0."
