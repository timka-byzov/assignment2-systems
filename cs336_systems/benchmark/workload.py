from abc import ABC, abstractmethod


class Workload(ABC):
    @abstractmethod
    def forward(self):
        pass

    @abstractmethod
    def backward(self):
        pass

    @abstractmethod
    def optimizer_step(self):
        pass
