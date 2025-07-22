import abc
from typing import TypeAlias, Self
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

    params: eqx.AbstractVar[Params | tuple[Params]]  # should have AnyParams here?

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


class AbstractJointDistribution(eqx.Module):
    """ """

    prior_model: eqx.AbstractVar[AbstractPriorDistribution]
    condition_llkh_model: eqx.AbstractVar[AbstractConditionalLikelihoodDistribution]

    def set_params(self, params: tuple[Params, Params]):
        new = eqx.tree_at(
            lambda pt: (pt.prior_model, pt.condition_llkh_model),
            self,
            (
                self.prior_model.set_params(params[0]),
                self.condition_llkh_model.set_params(params[1]),
            ),
        )
        return new

    @abc.abstractmethod
    def estimate_parameters(
        self,
        prior_realization: Array,
        condition_llkh_realization: Array,
        *args,
        **kwargs,
    ) -> tuple[Params, Params]:
        prior_params = self.prior_model.estimate_parameters(
            prior_realization, *args, **kwargs
        )
        condition_llkh_params = self.condition_llkh_model.estimate_parameters(
            condition_llkh_realization, prior_realization, *args, **kwargs
        )
        return prior_params, condition_llkh_params


class AbstractPosteriorDistribution(AbstractDistribution):
    """

    !!! warning
        The Params object of an AbstractPosteriorDistribution must reflect the
        order of the Params object of the prior first and of the
        condition_llkh second. The ordering must be respected inside the Params
        object too!!!
    """

    @abc.abstractmethod
    def estimate_parameters(
        self, self_realization: Array, obs_realization: Array, *_, **__
    ) -> Params:
        raise NotImplementedError
