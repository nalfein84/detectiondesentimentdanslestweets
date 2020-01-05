# -*- encoding: utf-8 -*-
from Normalisation import Normalise_Json
from Normalisation import Normalise_DonneeTest
from SVMscript import LaunchSVM
from projectHelper import PredictionToOut

def SVMNormalisation():
    # Normalise les données du TP
    Normalise_Json()
    # Normalise les données de Testes
    Normalise_DonneeTest()

def main():
    SVMNormalisation()
    fichierResultat = LaunchSVM("JSONdata", "texteDT", nameModel="SimpleModelFromHand")
    PredictionToOut(fichierResultat)
    
if __name__ == "__main__":
    # execute only if run as a script
    main()
