"""Containers for time series of conditional covariance matrices."""

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd


@dataclass
class CovarianceResult:
    """Time series of covariance matrices for one or more risk factors.

    Parameters
    ----------
    covariances : np.ndarray
        Covariance matrices with shape (T, N, N), including when N = 1.
    dates : pd.Index
        One label per matrix. For out-of-sample forecasts, these are the
        forecast *target* dates.
    assets : pd.Index
        Risk-factor labels, in the same order as the covariance axes.
    model : str, optional
        Name of the model used to produce the matrices.
    metadata : dict, optional
        Additional model information.

    Notes
    -----
    Matrices may be entirely NaN during initialization or burn-in. Partly
    missing matrices and infinite entries are not allowed. The class checks
    shapes, ordering, symmetry and nonnegative diagonal entries, but does
    not perform a (potentially expensive) positive-semidefiniteness check.

    The class does not distinguish fitted in-sample covariance paths from
    genuine out-of-sample forecasts. The producing function must document
    this distinction.
    """

    covariances: np.ndarray
    dates: pd.Index
    assets: pd.Index

    model: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.covariances = np.asarray(self.covariances, dtype=float)
        self.dates = pd.Index(self.dates)
        self.assets = pd.Index(self.assets)

        if self.covariances.ndim != 3:
            raise ValueError("covariances must have shape (T, N, N).")

        t = len(self.dates)
        n = len(self.assets)
        if t == 0 or n == 0:
            raise ValueError("dates and assets must not be empty.")
        if self.covariances.shape != (t, n, n):
            raise ValueError(
                "covariances shape must be "
                f"({t}, {n}, {n}), got {self.covariances.shape}."
            )
        if not self.dates.is_unique:
            raise ValueError("dates must be unique.")
        if not self.dates.is_monotonic_increasing:
            raise ValueError("dates must be sorted in increasing order.")
        if not self.assets.is_unique:
            raise ValueError("assets must be unique.")

        # A missing matrix is acceptable during initialization/burn-in.
        finite_matrices = np.isfinite(self.covariances).all(axis=(1, 2))
        missing_matrices = np.isnan(self.covariances).all(axis=(1, 2))
        if not np.all(finite_matrices | missing_matrices):
            raise ValueError(
                "Each covariance matrix must be fully finite or fully NaN."
            )

        valid = self.covariances[finite_matrices]
        if not np.allclose(valid, valid.swapaxes(1, 2), rtol=1e-10, atol=1e-12):
            raise ValueError("Covariance matrices must be symmetric.")
        if np.any(np.diagonal(valid, axis1=1, axis2=2) < -1e-12):
            raise ValueError("Covariance matrix diagonal entries cannot be negative.")

    @property
    def variances(self) -> pd.DataFrame:
        """Diagonal variances for all assets, indexed by date."""
        return pd.DataFrame(
            np.diagonal(self.covariances, axis1=1, axis2=2),
            index=self.dates,
            columns=self.assets,
        )

    def variance(self, asset: str) -> pd.Series:
        """Conditional variance history for one asset."""
        return self.variances[asset].rename(asset)

    def covariance(self, asset1: str, asset2: str) -> pd.Series:
        """Conditional covariance history for a pair of assets."""
        i = self.assets.get_loc(asset1)
        j = self.assets.get_loc(asset2)
        return pd.Series(
            self.covariances[:, i, j],
            index=self.dates,
            name=f"covariance({asset1}, {asset2})",
        )

    def correlation(self, asset1: str, asset2: str) -> pd.Series:
        """Conditional correlation; undefined if either variance is zero."""
        cov = self.covariance(asset1, asset2)
        v1 = self.variance(asset1)
        v2 = self.variance(asset2)
        with np.errstate(divide="ignore", invalid="ignore"):
            rho = cov / np.sqrt(v1 * v2)
        return rho.rename(f"correlation({asset1}, {asset2})")

    def at(self, date: Any) -> pd.DataFrame:
        """Covariance matrix at a specified date, with asset labels."""
        i = self.dates.get_loc(date)
        return pd.DataFrame(
            self.covariances[i],
            index=self.assets,
            columns=self.assets,
        )

    def latest(self) -> pd.DataFrame:
        """Latest covariance matrix, with asset labels."""
        return self.at(self.dates[-1])


# TEMPORARY COMPATIBILITY CLASS.
# Remove after models.py and rolling_forecast.py have migrated to CovarianceResult.
# @dataclass
# class VarianceResult:
#     """Legacy container for univariate variance histories."""

#     variances: pd.Series
#     asset: str

#     model: str | None = None
#     metadata: dict[str, Any] = field(default_factory=dict)






    
# from dataclasses import dataclass, field
# from typing import Any
# import pandas as pd
# import numpy as np

# @dataclass
# class CovarianceResult:
#     """Time series of covariance matrices."""

#     covariances: np.ndarray
#     dates: pd.Index
#     assets: pd.Index

#     model: str | None = None
#     metadata: dict[str, Any] = field(default_factory=dict)


# @dataclass
# class VarianceResult:
#     """Time series of variances."""

#     variances: pd.Series
#     asset: str

#     model: str | None = None
#     metadata: dict[str, Any] = field(default_factory=dict)
