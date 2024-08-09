from sample_factory.algo.runners.runner import AlgoObserver, Runner
from retinal_rl.rl.sample_factory.models import SampleFactoryBrain
import torch
import warnings

class SetWeightsObserver(AlgoObserver):
    """
    Observer runs on_init of the runner to set the model weights.
    Obviously only works if the brain passed here is the same as the one instantiated in the runner.
    """

    def __init__(self, sf_brain: SampleFactoryBrain):
        self.brain = sf_brain

    def on_start(self, runner: Runner) -> None:
            for learner_worker in runner.learners.values():
                try:
                    brain = learner_worker.param_server.actor_critic
                    overwrite_model_weights(brain, self.brain)
                except:
                     warnings.warn("Can not set model weights. Maybe ActorCritic is not a SFBrain?")

def overwrite_model_weights(model_to: torch.nn.Module, model_from: torch.nn.Module):
    with torch.no_grad():
        for [(_, to_param), (_, from_param)] in zip(model_to.named_parameters(),model_from.named_parameters()):
            to_param.data = from_param.data