from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

import models



@dataclass
class ForecastResult:
    covariance: np.ndarray
    origin_date: pd.Timestamp
    assets: pd.Index

    model: str
    metadata: dict[str, Any] = field(default_factory=dict)


class VolatilityForecaster(ABC):

    @abstractmethod
    def forecast(
        self,
        window: pd.DataFrame,
    ) -> ForecastResult:
        pass

class EWMAForecaster(VolatilityForecaster):

    def __init__(
        self,
        decay=0.96,
        initialization_window=60,
        burn_in_window=30,
    ):
        self.decay = decay
        self.initialization_window = initialization_window
        self.burn_in_window = burn_in_window

    def forecast(
        self,
        window: pd.DataFrame,
    ) -> ForecastResult:

        if window.shape[1] == 1:

            result = models.univariate_ewma_variance_estimator(
                window,
                decay=self.decay,
                initialization_window=self.initialization_window,
                burn_in_window=self.burn_in_window,
            )

            forecast = models.univariate_ewma_forecast(
                window,
                result,
            )

            covariance = np.array([
                [forecast.iloc[0]]
            ])

        else:

            result = models.multivariate_ewma_covariance_estimator(
                window,
                decay=self.decay,
                initialization_window=self.initialization_window,
                burn_in_window=self.burn_in_window,
            )

            forecast = models.multivariate_ewma_forecast(
                window,
                result,
            )

            covariance = forecast.to_numpy()

        return ForecastResult(
            covariance=covariance,
            origin_date=window.index[-1],
            assets=window.columns,
            model="EWMA",
            metadata={
                "decay": self.decay,
            },
        )



class GARCHForecaster(VolatilityForecaster):

    def __init__(
        self,
        p=1,
        q=1,
        o=0,
        dist="Normal",
    ):
        self.p = p
        self.q = q
        self.o = o
        self.dist = dist

    def forecast(
        self,
        window: pd.DataFrame,
    ) -> ForecastResult:

        if window.shape[1] != 1:
            raise ValueError(
                "GARCHForecaster currently supports one risk factor."
            )

        fit_result = models.fit_garch(
            window,
            p=self.p,
            q=self.q,
            o=self.o,
            dist=self.dist,
        )

        forecast = models.garch_forecast(
            fit_result
        )

        covariance = np.array([
            [forecast.iloc[0]]
        ])

        distribution = fit_result.model.distribution
        param_names = distribution.parameter_names()

        dist_params = (
            fit_result.params[param_names].to_dict()
            if param_names
            else None
        )

        return ForecastResult(
            covariance=covariance,
            origin_date=window.index[-1],
            assets=window.columns,
            model="GARCH",
            metadata={
                "p": self.p,
                "q": self.q,
                "o": self.o,
                "distribution": self.dist,
                "dist_params": dist_params,
            },
        )