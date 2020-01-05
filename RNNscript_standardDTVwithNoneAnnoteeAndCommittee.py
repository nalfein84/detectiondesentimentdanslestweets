# -*- encoding: utf-8 -*-
from Normalisation import Normalise_DonneeTest
from Normalisation import Normalise_Json
from projectHelper import PredictionToOut
from projectHelper import MakeFileWithUniqueValue
from Normalisation import Normalise_Unlabeled
from RNNscript import InitDocToVecParameter
from RNNscript import RNNscript

def RNN_Normalisation():
    # Normalise les données de Testes
    Normalise_DonneeTest()
    Normalise_Unlabeled()
    Normalise_Json()
    


def main():
    #RNN_Normalisation()
    InitDocToVecParameter(["JSONdata", "texteDT", "70kTweet"], 45, 10, 95, TestingDocToVec=True)
    RNNscript("JSONdata", "texteDT", 90, epochs=100, nameModel="standardsWithAnnotedTrainDataDTVwithNoneAnnoted", typeRNN="Committee", TailleDuComitee=20)
    PredictionToOut("GenerationFichier/RNNtestResult.txt")

if __name__ == "__main__":
    # execute only if run as a script
    main()
