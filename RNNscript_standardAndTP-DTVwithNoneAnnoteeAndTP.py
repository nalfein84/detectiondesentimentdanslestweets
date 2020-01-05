# -*- encoding: utf-8 -*-
from Normalisation import Normalise_DonneeTest
from Normalisation import Normalise_Json
from Normalisation import Normalise_Unlabeled
from Normalisation import Normalise_DataDeft2017
from projectHelper import ConcatFile
from projectHelper import PredictionToOut
from projectHelper import MakeFileWithUniqueValue
from RNNscript import InitDocToVecParameter
from RNNscript import RNNscript

def RNN_Normalisation():
    # Normalise les données de Testes
    Normalise_DonneeTest()
    Normalise_DataDeft2017()
    Normalise_Unlabeled()
    Normalise_Json()
    ConcatFile("Normalisation/JSONdata.txt", "Normalisation/SVMtrain.txt", "Normalisation/AnnotedAndTPdata.txt", withLabel=True)

def main():
    RNN_Normalisation()
    InitDocToVecParameter(["AnnotedAndTPdata", "texteDT", "70kTweet"], 100, 10, 95, TestingDocToVec=True)
    RNNscript("AnnotedAndTPdata", "texteDT", 95, epochs=5, nameModel="standardsWithAnnotedAndTP-DTVwithNoneAnnoted")
    PredictionToOut("GenerationFichier/RNNtestResult.txt")

if __name__ == "__main__":
    main()
