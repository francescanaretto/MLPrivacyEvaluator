
from PrivacyAttacks.PrivacyAttack import *


class AloaPrivacyAttack(PrivacyAttack):
    def __init__(self, black_box, attack_dataset_kind='stat', attack_dataset):
        super(black_box)
        self.attack_dataset = attack_dataset #noise, stat, random
        #qui controllo se stringa devo creare il dataset, se è dato lo uso e basta
        #stat, data
        self.train_set_attack = ...


    #impletare i due metodi astratti
    train(self.train_set_attack)

