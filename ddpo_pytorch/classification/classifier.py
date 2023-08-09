from abc import ABC, abstractmethod

class Classifier(ABC):

  @abstractmethod  
  def __init__(self):
    pass

  @abstractmethod
  def predict(self, x):
    pass

  @abstractmethod
  def preprocess(self, x):
    pass
