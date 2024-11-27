import torch
import numpy as np
from hydra.utils import instantiate
from omegaconf import DictConfig

from retinal_rl.classification.loss import ClassificationContext
from retinal_rl.models.objective import Objective
from runner.util import create_brain

def test_objective(classification_config: DictConfig):
    brain = create_brain(classification_config.brain)
    # brain.train()
    objective: Objective[ClassificationContext] = instantiate(classification_config.optimizer.objective, brain=brain)
    assert isinstance(objective, Objective)

    # create some random tensors to have sth for in / output
    input = torch.zeros(1,*brain.sensors['vision'])
    label = torch.ones(1,10)/10

    # forward pass
    stimuli = {'vision': input}
    responses = brain(stimuli)

    def grad_sum(model:torch.nn.Module):
        return np.sum([0 if param.grad is None else np.sum(param.grad.numpy()) for param in model.parameters()])

    assert grad_sum(brain) == 0

    context = ClassificationContext(
        sources=input,
        inputs=input,
        classes=label,
        responses=responses,
        epoch=1
    )
    loss_dict = objective.backward(context)
    assert loss_dict['percent_correct'] == 0 or loss_dict['percent_correct'] == 1

    assert grad_sum(brain) != 0, "Objective should change the gradients, but it's still 0."
