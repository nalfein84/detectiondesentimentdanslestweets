# -*- encoding: utf-8 -*-
from Normalisation import Normalise_DonneeTest
from Normalisation import Normalise_Json
from projectHelper import PredictionToOut
from projectHelper import MakeFileWithUniqueValue
from Normalisation import Normalise_Unlabeled
from Normalisation import ChargementMotsGrossier
from RNNscript import InitDocToVecParameter
from RNNscript import RNNscript

def RNN_Normalisation():
    # Normalise les données de Testes
    ChargementMotsGrossier()
    Normalise_DonneeTest(TypeNormalisation="Complete")
    Normalise_Unlabeled(TypeNormalisation="Complete")
    Normalise_Json(TypeNormalisation="Complete")
    


def main():
    RNN_Normalisation()
    InitDocToVecParameter(["JSONdata", "texteDT", "70kTweet"], 100, 10, 95, TestingDocToVec=True)
    RNNscript("JSONdata", "texteDT", 90, epochs=40 ,nameModel="standardsWithAnnotedTrainDataDTVwithNoneAnnotedAndWithoutBW")
    PredictionToOut("GenerationFichier/RNNtestResult.txt")

if __name__ == "__main__":
    # execute only if run as a script
    main()
