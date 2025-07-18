import abc
from typing import TypeAlias, Any, Self
from jaxtyping import Array, Key
import equinox as eqx

Params: TypeAlias = eqx.Module


class AbstractDistribution(eqx.Module):
    """
    Note that properties are not possible because the way to go for properties
    in dataclasses is using init=False, but this is not good to set init=False
    for eqx.Module wrt which we want to take gradients
    (https://github.com/patrick-kidger/equinox/issues/1034)
    """

    params: eqx.AbstractVar[Any]  # should have AnyParams here?

    def set_params(self, params: Params) -> Self:
        """
        About the order of fields (from dataclasses doc):
        The order of the fields in all of the generated methods is the order in which they appear in the class definition.
        """
        new = eqx.tree_at(lambda pt: pt.params, self, params)
        return new

    @abc.abstractmethod
    def sample(self, *_, key: Key, **__) -> Array:
        raise NotImplementedError


class AbstractPriorDistribution(AbstractDistribution):
    """
    For discrete distributions only (currently)
    """

    K: eqx.AbstractVar[int]

    @abc.abstractmethod
    def estimate_parameters(self, self_realization: Array, *_, **__) -> Params:
        raise NotImplementedError


class AbstractConditionalLikelihoodDistribution(AbstractDistribution):
    """ """

    prior_model: eqx.AbstractVar[AbstractPriorDistribution]

    @abc.abstractmethod
    def estimate_parameters(
        self, self_realization: Array, prior_realization: Array, *_, **__
    ) -> Params:
        raise NotImplementedError


class AbstractJointDistribution(AbstractDistribution):
    """ """

    prior_model: eqx.AbstractVar[AbstractPriorDistribution]
    condition_llkh_model: eqx.AbstractVar[AbstractConditionalLikelihoodDistribution]

    @abc.abstractmethod
    def estimate_parameters(
        self,
        prior_realization: Array,
        condition_llkh_realization: Array,
        *args,
        **kwargs,
    ) -> Params:
        prior_params = self.prior_model.estimate_parameters(
            prior_realization, *args, **kwargs
        )
        condition_llkh_params = self.condition_llkh_model.estimate_parameters(
            condition_llkh_realization, prior_realization, *args, **kwargs
        )
        self.params = tuple(*prior_params, *condition_llkh_params)
        return self.params


class AbstractPosteriorDistribution(AbstractDistribution):
    """ """

    @abc.abstractmethod
    def estimate_parameters(
        self, self_realization: Array, obs_realization: Array, *_, **__
    ) -> Params:
        raise NotImplementedError
