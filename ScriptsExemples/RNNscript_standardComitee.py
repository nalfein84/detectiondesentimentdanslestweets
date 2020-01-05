# -*- encoding: utf-8 -*-
from Normalisation import Normalise_DonneeTest
from Normalisation import Normalise_Json
from projectHelper import PredictionToOut
from RNNscript import InitDocToVecParameter
from RNNscript import RNNscript

def RNN_Normalisation():
    # Normalise les données de Testes
    Normalise_DonneeTest()
    Normalise_Json()

def main():
    RNN_Normalisation()
    InitDocToVecParameter(["JSONdata", "texteDT"], 100, 20, 98, TestingDocToVec=True)
    RNNscript("JSONdata", "texteDT", 80, epochs=100, nameModel="standardsWithAnnotedTrainDataAndCommitee", typeRNN="Committee", TailleDuComitee=10)
    PredictionToOut("GenerationFichier/RNNtestResult.txt")

if __name__ == "__main__":
    # execute only if run as a script
    main()
