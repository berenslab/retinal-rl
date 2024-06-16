from retinal_rl_models.base_model import BaseModel
from torch import nn
from collections import OrderedDict
import os

class AssembledModel(BaseModel):
    """
    This model is intended to be used as the "overarching model" combining several "partial models" - such as encoders, latents and decoders - in a sequential way.
    """
    def __init__(self, partial_models: list[BaseModel|dict]) -> None:
        """
        partial_models: list of either objects of type BaseModel or dictionaries defining the individual models
        """
        model_configs = []
        models = OrderedDict()
        for m in partial_models:
            if isinstance(m, BaseModel):
                model_configs.append(m.config)
                models[m.config["type"]] = m
            elif isinstance(m, dict):
                model_configs.append(m)
                models[m["type"]] = BaseModel.model_from_config(m)
            else:
                raise TypeError("partial_models can only contain objects of type BaseModel or dict")
        super().__init__({"partial_models": model_configs})

        self.partial_models = nn.Sequential(models)

        #TODO: add assertions to make sure models are "compatible" (dimensionality etc)

    def forward(self, x):
        return self.partial_models(x)
    
    def save(self, filename, save_cfg=True, unassembled=False):
        if not unassembled:
            super().save(filename, save_cfg)
        else:
            if not os.path.exists(filename):
                os.mkdir(filename)
            for model in self.partial_models: # TODO: Implement loading from folder?
                model_path = os.path.join(filename, model.config["type"])
                model.save(model_path, save_cfg)