"""Statistical helper functions for uncertainty and sigma reports."""


from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Optional, Tuple

def binom_se(p: float, n: int) -> float:

    n = int(n)
    if n <= 0:
        return float("nan")
    p = float(p)
    v = max(0.0, p * (1.0 - p))
    return math.sqrt(v / n)

@dataclass(frozen=True)
class SigmaReport:
    ref: float
    ref_se: float
    exp: float
    exp_se: Optional[float]
    abs_diff: float
    rel_diff: float
    z_sigma: float
    pass_3sigma: bool

def sigma_report(exp: float, ref: float, ref_se: float, exp_se: Optional[float] = None, eps: float = 1e-15) -> SigmaReport:

    exp = float(exp); ref = float(ref); ref_se = float(ref_se)
    abs_diff = abs(exp - ref)
    denom_ref = max(abs(ref), eps)
    rel_diff = abs_diff / denom_ref

    denom_se = max(ref_se, eps)
    z = abs_diff / denom_se
    return SigmaReport(
        ref=ref,
        ref_se=ref_se,
        exp=exp,
        exp_se=None if exp_se is None else float(exp_se),
        abs_diff=abs_diff,
        rel_diff=rel_diff,
        z_sigma=z,
        pass_3sigma=(z < 3.0),
    )
