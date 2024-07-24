import torch

class OutputHook(list[torch.Tensor]):
    """ Hook to capture module outputs.
    """
    def __call__(self, module: torch.nn.Module, input:torch.Tensor, output:torch.Tensor):
        self.append(output)

def l1reg(x:torch.Tensor):
    return torch.abs(x).sum()

def l2reg(x:torch.Tensor):
    return torch.pow(x, 2).sum()

class ActivationRegularization(): # TODO: make less memory demanding
    def __init__(self, module: torch.nn.Module | list[torch.nn.Module], p:int=2, act_lambda:float=0.01):
        self._lambda = act_lambda
        if act_lambda > 0:
            self.hook = OutputHook()
            if p == 1:
                self.norm = l1reg
            else:
                self.norm = l2reg
            if isinstance(module, torch.nn.Module):
                module.register_forward_hook(self.hook)
            else:
                for m in module:
                    m.register_forward_hook(self.hook)

    def penalty(self):
        penalty = 0.
        if self._lambda > 0:
            penalty = self._lambda * sum(self.norm(output) for output in self.hook)
            self.hook.clear()
        return penalty

class WeightRegularization():
    def __init__(self, module: torch.nn.Module | list[torch.nn.Module], p:int=2, weight_decay:float=0.01):
        self._lambda = weight_decay
        if self._lambda > 0:
            if p == 1:
                self.norm = l1reg
            else:
                self.norm = l2reg
            self.module = module

    def penalty(self):
        penalty = 0.
        if self._lambda > 0:
            if isinstance(self.module, torch.nn.Module):
                penalty = self._lambda * sum(self.norm(param) for param in self.module.parameters())
            else:
                penalty = self._lambda * sum(self.norm(param) for module in self.module for param in module.parameters())
        return penalty