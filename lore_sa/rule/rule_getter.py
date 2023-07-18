from abc import ABC, abstractmethod

from lore_sa.dataset import TabularDataset
from lore_sa.encoder_decoder import EncDec
from lore_sa.surrogate.surrogate import Surrogate

__all__ = ["Emitter"]

class Emitter(ABC):

    """
    Interfaces to get the rules and the counterfactual rules from a surrogate model
    """

    @abstractmethod
    def get_rule(self,x, y, dt: Surrogate, dataset: TabularDataset, encdec: EncDec = None, multi_label: bool = False):
        pass

    @abstractmethod
    def get_counterfactual_rules(self,x, y):
        pass