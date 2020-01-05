# -*- encoding: utf-8 -*-
from Normalisation import Normalise_DonneeTest
from Normalisation import Normalise_Json
from projectHelper import PredictionToOut
from projectHelper import MakeFileWithUniqueValue
from Normalisation import EnleverMotVideSpecifique
from Normalisation import ChargementMotsGrossier
from Normalisation import Normalise_Unlabeled
from Normalisation import ChargementMotsVide
from RNNscript import InitDocToVecParameter
from RNNscript import RNNscript

def RNN_Normalisation():
    # Normalise les données de Testes
    #ChargementMotsVide()
    #ChargementMotsGrossier()
    #Normalise_DonneeTest(TypeNormalisation="Complete")
    #Normalise_Unlabeled(TypeNormalisation="Complete")
    #Normalise_Json(TypeNormalisation="Complete")
    #EnleverMotVideSpecifique("JSONdata")
    #EnleverMotVideSpecifique("texteDT")
    #EnleverMotVideSpecifique("70kTweet")
    print("odfejzpo")


def main():
    RNN_Normalisation()
    InitDocToVecParameter(["JSONdata", "texteDT", "70kTweet"], 50, 10, 90, TestingDocToVec=True)
    RNNscript("JSONdata", "texteDT", 90, epochs=200, SelectBestModel=True, typeRNN="Committee" ,TailleDuComitee=20 ,nameModel="standardsWithAnnotedTrainDataDTVwithNoneAnnotedAndWithoutEW")
    PredictionToOut("GenerationFichier/RNNtestResult.txt")

if __name__ == "__main__":
    # execute only if run as a script
    main()
