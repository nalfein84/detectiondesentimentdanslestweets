# -*- encoding: utf-8 -*-
from Normalisation import Normalise_Json
from Normalisation import Normalise_DonneeTest
from Normalisation import ChargementMotsGrossier
from Normalisation import ChargementMotsVide
from SVMscript import LaunchSVM
from projectHelper import PredictionToOut

def SVMNormalisation():
    ChargementMotsGrossier()
    ChargementMotsVide()
    # Normalise les données du TP
    Normalise_Json(TypeNormalisation="Complete")
    # Normalise les données de Testes
    Normalise_DonneeTest(TypeNormalisation="Complete")

def main():
    SVMNormalisation()
    fichierResultat = LaunchSVM("JSONdata", "texteDT", nameModel="ModelFromHandWithoutBWandEW")
    PredictionToOut(fichierResultat)
    
if __name__ == "__main__":
    main()
