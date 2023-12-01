from abc import ABC, abstractmethod
from MLWrapper.bbox import AbstractBBox

class PrivacyAttack(ABC):
    def __init__(self, black_box: AbstractBBox):
        pass
        self.bb = black_box

    @abstractmethod
    def PrivacyAttackTrain(self, train_set_attack, test_set_attack, train_label_attack...):
        pass

    @abstractmethod
    def PrivacyAttackTest(self):
        pass


