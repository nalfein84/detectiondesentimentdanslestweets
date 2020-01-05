# -*- encoding: utf-8 -*-
from Normalisation import Normalise_DataDeft2017
from Normalisation import Normalise_DonneeTest
from SVMscript import LaunchSVM
from projectHelper import PredictionToOut
from Normalisation import EnleverMotVideSpecifique
from Normalisation import ChargementMotsGrossier
from Normalisation import ChargementMotsVide

def SVMNormalisation():
    ChargementMotsGrossier()
    ChargementMotsVide()
    
    # Normalise les données du TP 
    Normalise_DataDeft2017(TypeNormalisation="Complexe")
    # Normalise les données de Testes
    Normalise_DonneeTest(TypeNormalisation="Complexe")
    EnleverMotVideSpecifique("SVMtrain")
    EnleverMotVideSpecifique("SVMtest")
    EnleverMotVideSpecifique("texteDT")

def main():
    SVMNormalisation()
    fichierResultat = LaunchSVM("SVMtrain", "texteDT", nameTest="SVMtest", nameModel="ModelFromTPDataWithoutBW-and-EW")
    PredictionToOut(fichierResultat)
    
if __name__ == "__main__":
    # execute only if run as a script
    main()
