from abc import ABC, abstractmethod
class BaseModel(ABC):
    @abstractmethod
    def fit(self, train):
        pass
    @abstractmethod
    def predict(self, steps):
        pass