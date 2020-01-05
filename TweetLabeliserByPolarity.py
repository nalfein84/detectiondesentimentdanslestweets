from Normalisation import CorpusLabelWordToPolarity
from projectHelper import txtFileToListe
from projectHelper import WriteInFile
from Normalisation import Normalise_DonneeTest
from projectHelper import GetIndexFromNormaliseFile
from SVMscript import TXTtoSVM
from SVMscript import DataTestToSMV
import subprocess

WordPolarity = {}


def PolarityFromValues(tab, mot):
    if tab[0] + tab[1] + tab[2] <= 10:
        return [0,0,0]
    if tab[2] == 0 and (float(tab[1])/float(tab[0] + tab[1] + tab[2])) > 0.3:
        return [0,1,0]
    maxV = tab[0]
    IDmax = 0
    res = [0,0,0]
    id = 0
    for value in tab:
        if maxV < value:
            maxV = value
            IDmax = id
        id += 1
    res[IDmax] = 1
    return res

def GetPolarityByWord():
    global WordPolarity
    dataWordPolarity = {}
    listePolarite = txtFileToListe("Normalisation/Polarite.txt")
    for polarite in listePolarite:
        polarite = polarite.replace("\n", '')
        data = polarite.split(";")
        mot = data[0]
        polariteMot = [int(data[1]), int(data[2]), int(data[3])]
        if dataWordPolarity.has_key(mot):
            polariteDejaExistante = dataWordPolarity[mot]
            dataWordPolarity[mot] = [polariteDejaExistante[0] + polariteMot[0], polariteDejaExistante[1] + polariteMot[1], polariteDejaExistante[2] + polariteMot[2]]
        else:
            dataWordPolarity[mot] = polariteMot
    listeMotGrossier = txtFileToListe("Normalisation/MG.txt")
    for grossiertee in listeMotGrossier:
        WordPolarity[grossiertee] = [0, 0, 1]
    
    for mot in dataWordPolarity.keys():
        WordPolarity[mot] = PolarityFromValues(dataWordPolarity[mot], mot)
    


def UnlabeledTweetToPolarity():
    global WordPolarity
    listeTweet = txtFileToListe("Normalisation/70kTweet.txt")
    nbrBad = 0
    nbrCool = 0
    nbrNormal = 0
    nbrMixte = 0
    listTweetLabelliser = []
    listTweetLabel = []
    for tweet in listeTweet:
        tweet.replace("\n", '')
        polarite = [0,0,0]
        for mot in tweet.split():
            if WordPolarity.has_key(mot):
                polarite = [polarite[0] + WordPolarity[mot][0], polarite[1] + WordPolarity[mot][1], polarite[2] + WordPolarity[mot][2]]
        
        if polarite[2] > 2:
            #print("Mechant : " + str(polarite) + " || " + tweet)
            listTweetLabelliser.append(tweet + "\n")
            listTweetLabel.append("3" + "\n")
            nbrBad += 1
        elif polarite[0] > 5 and polarite[2] == 0:
            #print("Gentil : " + str(polarite) + " || " + tweet)
            listTweetLabelliser.append(tweet + "\n")
            listTweetLabel.append("2" + "\n")
            nbrCool += 1
        elif polarite[0] == 0 and polarite[2] == 0 and polarite[1]>3:
            print("Neutre : " + str(polarite) + " || " + tweet)
            listTweetLabelliser.append(tweet + "\n")
            listTweetLabel.append("0" + "\n")
            nbrNormal += 1
    
    #WriteInFile("Normalisation/SVMunlabeled.txt", listTweetLabelliser)
    #WriteInFile("Normalisation/SVMunlabeledlabel.txt", listTweetLabel)
    print("Bad = " + str(nbrBad) + "| Cool = " + str(nbrCool) + "| neutre = " + str(nbrNormal) + "| mixte = " + str(nbrMixte))

def LancementSVM():
    Normalise_DonneeTest(TypeNormalisation="Complexe")
    index = GetIndexFromNormaliseFile("Normalisation/texteDT.txt", GetIndexFromNormaliseFile("Normalisation/SVMunlabeled.txt"))
    DataTestToSMV(index)
    TXTtoSVM("Normalisation/SVMunlabeled.txt", "GenerationFichier/TweetUnlabeled.svm", index)
    subprocess.call(["liblinear-2.30/train", "-c", "1", "-e", "0.1", "GenerationFichier/TweetUnlabeled.svm", "GenerationFichier/tweets.model"])

    print("Lancement de la prediction sur le sujet sans annotation")
    # Lancement de la prediction sur le sujet sans annotation :
    subprocess.call(["liblinear-2.30/predict", "GenerationFichier/SVM/DT.svm",
                    "GenerationFichier/tweets.model", "GenerationFichier/resultModel.txt"])
    


#Normalisation du corpus 
#CorpusLabelWordToPolarity()

#GetPolarityByWord()
#UnlabeledTweetToPolarity()
LancementSVM()
