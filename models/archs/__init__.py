from utils import Factory

arch_factory = Factory()

from .fcn import FCN, FCNActorCritic

__all__ = ["FCN", "FCNActorCritic"]
