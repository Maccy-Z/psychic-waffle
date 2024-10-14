from abc import ABC, abstractmethod
import torch

class BaseDerivCalc(ABC):
    @abstractmethod
    def derivative(self, Xs: torch.Tensor) -> dict[tuple, torch.Tensor]:
        pass
