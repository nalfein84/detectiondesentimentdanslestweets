# -*- encoding: utf-8 -*-
from Normalisation import Normalise_Json
from Normalisation import Normalise_DonneeTest
from Normalisation import Normalise_DataDeft2017
from SVMscript import LaunchSVM
from projectHelper import PredictionToOut
from projectHelper import ConcatFile

def SVMNormalisation():
    Normalise_Json()
    # Normalise les données du TP
    Normalise_DataDeft2017()
    # Normalise les données de Testes
    Normalise_DonneeTest()

def main():
    SVMNormalisation()
    ConcatFile("Normalisation/JSONdata.txt", "Normalisation/SVMtrain.txt", "Normalisation/TPandJson.txt", withLabel=True)
    fichierResultat = LaunchSVM("TPandJson", "texteDT", nameModel="SimpleModelFromHandAndTP")
    PredictionToOut(fichierResultat)
    
if __name__ == "__main__":
    # execute only if run as a script
    main()
