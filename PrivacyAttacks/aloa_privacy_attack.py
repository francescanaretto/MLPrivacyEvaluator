
from PrivacyAttacks.privacy_attack import *


class AloaPrivacyAttack(PrivacyAttack):
    def __init__(self, black_box, attack_dataset_kind='stat', attack_dataset):
        super.__init__(black_box)
        self.attack_dataset = attack_dataset #noise, stat, random
        #qui controllo se stringa devo creare il dataset, se è dato lo uso e basta
        #stat, data
        self.train_set_attack = ...


    #impletare i due metodi astratti
    def train(self.train_set_attack):
        pass
