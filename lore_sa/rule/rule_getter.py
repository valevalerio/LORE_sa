from abc import ABC, abstractmethod

from lore_sa.dataset import Dataset
from lore_sa.encoder_decoder import EncDec
from lore_sa.rule import Rule
from lore_sa.surrogate.surrogate import Surrogate

__all__ = ["Rule", "RuleGetter"]
class RuleGetter(ABC):

    """
    Interfaces to get the rules and the counterfactual rules
    """

    @abstractmethod
    def get_rule(self,x, y, dt: Surrogate, dataset: Dataset, encdec: EncDec = None, multi_label: bool = False):
        pass

    @abstractmethod
    def get_counterfactual_rules(self,x, y):
        pass