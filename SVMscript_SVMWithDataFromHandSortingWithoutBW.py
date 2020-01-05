# -*- encoding: utf-8 -*-
from Normalisation import Normalise_Json
from Normalisation import Normalise_DonneeTest
from Normalisation import ChargementMotsGrossier
from SVMscript import LaunchSVM
from projectHelper import PredictionToOut

def SVMNormalisation():
    ChargementMotsGrossier()
    # Normalise les données du TP
    Normalise_Json(TypeNormalisation="Complete")
    # Normalise les données de Testes
    Normalise_DonneeTest(TypeNormalisation="Complete")

def main():
    SVMNormalisation()
    fichierResultat = LaunchSVM("JSONdata", "texteDT", nameModel="ModelFromHandWithoutBW")
    PredictionToOut(fichierResultat)
    
if __name__ == "__main__":
    # execute only if run as a script
    main()
