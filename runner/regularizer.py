from typing import Callable
import torch

class OutputNormHook():
    def __init__(self, norm: Callable[[torch.Tensor], torch.Tensor]):
        """ Hook to capture module outputs. Sums them up according to the specified norm"""
        self.norm=norm
        self._val:torch.Tensor = torch.tensor(0.0)
    def __call__(self, module: torch.nn.Module, input:torch.Tensor, output:torch.Tensor):
        self._val += self.norm(output)

    def get(self) -> torch.Tensor:
        """Returns the value and resets the hook"""
        val = self._val
        self._val= torch.tensor(0.0)
        return val
    
def l1reg(x:torch.Tensor):
    return torch.abs(x).sum()

def l2reg(x:torch.Tensor):
    return torch.pow(x, 2).sum()

class ActivationRegularization():
    def __init__(self, module: torch.nn.Module | list[torch.nn.Module], p:int=2, act_lambda:float=0.01):
        """
        Add an activation regularizer to the module.
        Initialize with module, will keep track of all produced outputs and sum them up according to the specified norm.

        Parameters:
            module (nn.Module or list[nn.Module]): Module(s) to apply the regularizer to.
            p (int): The norm type. Use 1 for L1 norm and 2 for L2 norm.
            act_lambda (float): Regularization strength.
        """
        self._lambda = act_lambda
        if act_lambda > 0:
            if p == 1:
                self.norm = l1reg
            else:
                self.norm = l2reg
            self.hook = OutputNormHook(self.norm)
            if isinstance(module, torch.nn.Module):
                module.register_forward_hook(self.hook)
            else:
                for m in module:
                    m.register_forward_hook(self.hook)

    def penalty(self):
        penalty = 0.
        if self._lambda > 0:
            penalty = self._lambda * self.hook.get()
        return penalty

class WeightRegularization():
    def __init__(self, module: torch.nn.Module | list[torch.nn.Module], p:int=2, weight_decay:float=0.01):
        """
        Add a weight regularizer to the module.
        Initialize with module, will keep track of all weights and sum them up according to the specified norm.

        Parameters:
            module (nn.Module or list[nn.Module]): Module(s) to apply the regularizer to.
            p (int): The norm type. Use 1 for L1 norm and 2 for L2 norm.
            weight_decay (float): Regularization strength.
        """
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