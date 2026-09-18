
from dataclasses import dataclass, field
from typing import Any
import pandas as pd

@dataclass
class CovarianceResult:
    """Time series of covariance matrices."""

    covariances: pd.Series
    dates: pd.Index
    assets: pd.Index

    model: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class VarianceResult:
    """Time series of variances."""

    variances: pd.Series
    asset: str

    model: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
