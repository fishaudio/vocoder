from typing import Union

import torch
from torch import Tensor
from torch.utils._foreach_utils import (
    _group_tensors_by_device_and_dtype,
    _has_foreach_support,
)


@torch.no_grad()
def grad_norm(
    parameters: Union[Tensor, list[Tensor]],
    norm_type: float = 2.0,
) -> float:
    """
    Returns the norm of the gradients of the given parameters.

    Args:
        parameters (Iterable[Tensor] or Tensor): an iterable of Tensors or a
            single Tensor that will have gradients normalized
        norm_type (float): type of the used p-norm.

    Returns:
        Total norm of the parameter gradients (viewed as a single vector).
    """  # noqa: E501

    if isinstance(parameters, Tensor):
        parameters = [parameters]

    grads = [p.grad for p in parameters if p.grad is not None]
    first_device = grads[0].device
    grouped_grads: dict[
        tuple[torch.device, torch.dtype], list[list[Tensor]]
    ] = _group_tensors_by_device_and_dtype(
        [[g.detach() for g in grads]]
    )  # type: ignore[assignment]

    norms = []
    for (device, _), [grads] in grouped_grads.items():
        if _has_foreach_support(grads, device=device):
            norms.extend(torch._foreach_norm(grads, norm_type))
        else:
            norms.extend([torch.norm(g, norm_type) for g in grads])

    return torch.norm(torch.stack([norm.to(first_device) for norm in norms]), norm_type)
