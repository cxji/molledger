"""
Per-task label transforms.

Choice of transform per task lives on TaskSpec.label_transform:
  "none"  -> identity: already-log quantities (logd, aqueous_solubility/ESOL), binary (hia), and
             caco2_papp_ab (harmonized to log10 units before any transform here).
  "log"   -> natural log. Strictly-positive columns with a wide low-end dynamic range and no exact
             zeros (half_life, caco2_efflux_ratio, kinetic_solubility).
  "log1p" -> log(1+x). Heavy-tailed positive columns with exact zeros (both clint, the three ppb
             tasks): log1p(0) = 0 where log(0) would be -inf. Inverse is expm1.
"""

import torch

_FORWARD = {
    "none": lambda x: x,
    "log": torch.log,
    "log1p": torch.log1p,
}
_INVERSE = {
    "none": lambda x: x,
    "log": torch.exp,
    "log1p": torch.expm1,
}


def forward_transform(name: str, x: torch.Tensor) -> torch.Tensor:
    """Native -> model space. NaN passes through as NaN."""
    return _FORWARD[name](x)


def inverse_transform(name: str, x: torch.Tensor) -> torch.Tensor:
    """Model space -> native units (for reporting)."""
    return _INVERSE[name](x)
