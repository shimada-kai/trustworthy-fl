from abc import ABC, abstractmethod


class ModelPoisoningAttack(ABC):
    @abstractmethod
    def apply(self, update):
        raise NotImplementedError
