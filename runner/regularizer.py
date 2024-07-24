import torch

class OutputHook(list):
    """ Hook to capture module outputs.
    """
    def __call__(self, module, input, output):
        self.append(output)

def l1reg(x:torch.Tensor):
    return torch.abs(x).sum()

def l2reg(x:torch.Tensor):
    return torch.pow(x, 2).sum()

class ActivationRegularization(): # TODO: make less memory demanding
    def __init__(self, module = torch.nn.Module | list, p=2.0, act_lambda=0.01):
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
    def __init__(self, module = torch.nn.Module | list, p=2.0, weight_decay=0.01):
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
            penalty = self._lambda * sum(self.norm(param) for param in self.module.parameters())
        return penalty