import json
import logging
import re
from enum import Enum
from math import ceil, floor
from typing import Any, List, Optional, Tuple, TypeVar, Union, cast

import numpy as np
from numpy.typing import NDArray
from torch import nn
import torch

logger = logging.getLogger(__name__)

### Types

FloatArray = NDArray[np.float64]

T = TypeVar("T")

### IO Handling stuff


class NumpyEncoder(json.JSONEncoder):
    """JSON encoder that handles numpy arrays."""

    def default(self, obj: Any) -> Any:
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)


### Functions


def assert_list(
    list_candidate: Union[T, List[T]],
    len_list: int,
) -> List[T]:
    """Assert that the list has the expected length, or create a list of the same value repeated."""
    if isinstance(list_candidate, List):
        list_candidate = cast(List[T], list_candidate)
        if len(list_candidate) != len_list:
            raise AssertionError(
                f"The length of the list does not match the expected length: {len(list_candidate)} != {len_list}"
            )
        return list_candidate
    return [list_candidate] * len_list


def camel_to_snake(name: str) -> str:
    """Convert a CamelCase string to snake_case."""
    # This regex looks for uppercase letters
    # If it's at the start of the string, it just lowercase it
    # If it's preceded by a lowercase letter, it adds an underscore before it
    pattern = re.compile(r"(?<!^)(?=[A-Z])")
    return pattern.sub("_", name).lower()


def encoder_out_size(mdls: List[nn.Module], hght0: int, wdth0: int) -> Tuple[int, int]:
    """Compute the size of the encoder output, where mdls is the list of encoder modules."""
    hght = hght0
    wdth = wdth0

    # iterate over modules that are not activations
    for mdl in mdls:
        if is_nonlinearity(mdl):
            continue
        if isinstance(mdl, nn.Conv2d):
            krnsz = _double_up(mdl.kernel_size)
            strd = _double_up(mdl.stride)
            pad = _double_up(mdl.padding)
            dila = _double_up(mdl.dilation)
        else:
            raise NotImplementedError("Only convolutional layers are supported")

        if dila[0] != 1 or dila[1] != 1:
            raise NotImplementedError("Dilation not implemented")

        # if has a ceil mode
        if hasattr(mdl, "ceil_mode") and mdl.ceil_mode:
            hght = ceil((hght - krnsz[0] + 2 * pad[0]) / strd[0] + 1)
            wdth = ceil((wdth - krnsz[1] + 2 * pad[1]) / strd[1] + 1)
        else:
            hght = floor((hght - krnsz[0] + 2 * pad[0]) / strd[0] + 1)
            wdth = floor((wdth - krnsz[1] + 2 * pad[1]) / strd[1] + 1)

    return hght, wdth


def rf_size_and_start(
    mdls: List[nn.Module], hidx: int, widx: int
) -> Tuple[int, int, int, int]:
    """Compute the receptive field size and start position for sequence of modules."""
    hrf_size = 1
    hrf_scale = 1
    hrf_shift = 0

    wrf_size = 1
    wrf_scale = 1
    wrf_shift = 0

    hmn = hidx
    wmn = widx

    for mdl in mdls:
        if is_nonlinearity(mdl):
            continue
        if not (is_convolutional_layer(mdl) or is_base_pooling_layer(mdl)):
            raise NotImplementedError(
                "Only convolutional and basic pooling layers are supported"
            )

        hksz, wksz = _double_up(mdl.kernel_size)
        hstrd, wstrd = _double_up(mdl.stride)
        hpad, wpad = _double_up(mdl.padding)

        hrf_size += (hksz - 1) * hrf_scale
        wrf_size += (wksz - 1) * wrf_scale

        hrf_shift += hpad * hrf_scale
        wrf_shift += wpad * wrf_scale

        hrf_scale *= hstrd
        wrf_scale *= wstrd

        hmn = hidx * hrf_scale - hrf_shift
        wmn = widx * wrf_scale - wrf_shift

    return hrf_size, wrf_size, hmn, wmn


class Activation(Enum):
    elu = nn.ELU
    relu = nn.ReLU
    tanh = nn.Tanh
    softplus = nn.Softplus
    leaky = nn.LeakyReLU
    identity = nn.Identity
    gelu = nn.GELU
    sigmoid = nn.Sigmoid

    def __call__(self) -> nn.Module:
        act_module = self.value()
        if hasattr(act_module, "inplace"):
            act_module = self.value(inplace=True)
        return act_module


def is_nonlinearity(mdl: nn.Module) -> bool:
    """Check if the module is an activation function or a normalization layer."""
    return mdl.__class__ in [act.value for act in Activation] or isinstance(
        mdl,
        (
            nn.BatchNorm1d,
            nn.BatchNorm2d,
            nn.BatchNorm3d,
            nn.InstanceNorm1d,
            nn.InstanceNorm2d,
            nn.InstanceNorm3d,
        ),
    )


def is_convolutional_layer(mdl: nn.Module) -> bool:
    return isinstance(mdl, (nn.Conv1d, nn.Conv2d, nn.Conv3d))


def is_base_pooling_layer(mdl: nn.Module) -> bool:
    return isinstance(
        mdl,
        (
            nn.AvgPool1d,
            nn.AvgPool2d,
            nn.AvgPool3d,
            nn.MaxPool1d,
            nn.MaxPool2d,
            nn.MaxPool3d,
        ),
    )


def _double_up(x: Union[int, Tuple[int, ...]]):
    if isinstance(x, int):
        return (x, x)
    return x

ArrayLike = TypeVar("ArrayLike", np.ndarray, torch.Tensor)
def rescale_zero_one(x: ArrayLike, min: Optional[float] = None, max: Optional[float] = None) -> ArrayLike:
    if min is None:
        min = np.min(x) if isinstance(x, np.ndarray) else torch.min(x).item()
    if max is None:
        max = np.max(x) if isinstance(x, np.ndarray) else torch.max(x).item()
    return (x - min) / (max - min + 1e-8)