# This code is part of Qiskit.
#
# (C) Copyright IBM 2022.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Discriminator wrappers to make discriminators serializable.."""

from abc import abstractmethod
from typing import Any, Dict, List

from qiskit_experiments.data_processing.exceptions import DataProcessorError

try:
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False


class BaseDiscriminator:
    """An abstract base class for serializable discriminators.

    ``BaseDiscriminators`` are used in the :class:`.Discriminator` data action nodes.
    """

    @abstractmethod
    def predict(self, data):
        """The function used to predict the labels of the data."""

    @property
    def discriminator(self) -> Any:
        """Return the discriminator object that is wrapped.

        Sub-classes may not need to implement this method but can chose to
        if they are wrapping an object capable of discrimination.
        """
        return None

    @abstractmethod
    def config(self) -> Dict[str, Any]:
        """Return the configuration of the discriminator."""

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "BaseDiscriminator":
        """Create a discriminator from the configuration."""

    def __json_encode__(self):
        """Convert to format that can be JSON serialized."""
        return self.config()

    @classmethod
    def __json_decode__(cls, value: Dict[str, Any]) -> "BaseDiscriminator":
        """Load from JSON compatible format."""
        return cls.from_config(value)


class LDA(BaseDiscriminator):
    """A wrapper for the SKlearn linear discriminant analysis."""

    def __init__(self, lda: "LinearDiscriminantAnalysis"):
        """
        Args:
            lda: The sklearn linear discriminant analysis. This may be a trained or an
                untrained discriminator.
        """
        if not HAS_SKLEARN:
            raise DataProcessorError(
                f"SKlearn is needed to initialize an {self.__class__.__name__}."
            )

        self._lda = lda
        self.attributes = [
            "coef_",
            "intercept_",
            "covariance_",
            "explained_variance_ratio_",
            "means_",
            "priors_",
            "scalings_",
            "xbar_",
            "classes_",
            "n_features_in_",
            "feature_names_in_",
        ]

    @property
    def discriminator(self) -> Any:
        """Return then SKLearn object."""
        return self._lda

    def predict(self, data):
        """Wrap the predict method of the LDA."""
        return self._lda.predict(data)

    def fit(self, data: List, labels: List):
        """Fit the LDA.

        Args:
            data: The independent data.
            labels: The labels corresponding to data.
        """
        self._lda.fit(data, labels)

    def config(self) -> Dict[str, Any]:
        """Return the configuration of the LDA."""
        attr_conf = {attr: getattr(self._lda, attr, None) for attr in self.attributes}
        return {"params": self._lda.get_params(), "attributes": attr_conf}

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "LDA":
        """Deserialize from an object."""

        if not HAS_SKLEARN:
            raise DataProcessorError(f"SKlearn is needed to initialize an {cls.__name__}.")

        lda = LinearDiscriminantAnalysis()
        lda.set_params(**config["params"])

        for name, value in config["attributes"].items():
            if value is not None:
                setattr(lda, name, value)

        return LDA(lda)
