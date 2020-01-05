# -*- encoding: utf-8 -*-

from projectHelper import GetIndexFromNormaliseFile
from projectHelper import txtFileToListe
from IndexBuilder import IndexBuilder
import subprocess
import os
import operator

## Commande SVM:
# liblinear-2.30/train -c 1 -e 0.1 GenerationFichier/SVM/trainData.svm GenerationFichier/tweets.model
# liblinear-2.30/predict GenerationFichier/SVM/DT.svm GenerationFichier/tweets.model GenerationFichier/resultModel.txt
# 

def lineToSVMfile(svmFilename, line, label, index):
    value = str(label)

    mots = line.split()
    # Initialisation du dictionnaire contenant les mots de la phrase
    dicMot = {}.fromkeys(set(mots), 0)
    for mot in dicMot.keys():
            dicMot[mot] = [0, 0]
    # Récupération de l'index pour chaque mot
    for mot in dicMot.keys():
        dicMot[mot][0] = index.wordslist[mot][2] + 1
    # Comptage du nombre d'occurence pour chaque mot
    for valeur in mots:
        dicMot[valeur][1] += 1

    # Trie des mots en fonction de l'ID
    dico_trie = sorted(dicMot.iteritems(), reverse=False,
                        key=operator.itemgetter(1))
    
    # Création de la ligne SVM-compatible
    for element in dico_trie:
        value += " " + str(element[1][0]) + ":" + str(element[1][1])
    value += "\n"
    svmFile = open(svmFilename, 'a')
    svmFile.write(value)
    svmFile.close()

def DataPredictionToSMV(filename, index):
    svmFilename = "GenerationFichier/SVM/DT.svm"
    if os.path.isfile(svmFilename):
        os.remove(svmFilename)
    lignes = txtFileToListe("Normalisation/texteDT.txt")
    for line in lignes:
        lineToSVMfile(svmFilename, line, '0', index)
    return index


def TXTtoSVM(filename, svmFilename, index, labelFilename=None):
    if os.path.isfile(svmFilename):
        os.remove(svmFilename)
        
    print("Lancement de la transformation du fichier " + filename)
    lignes = txtFileToListe(filename)
    labels = []

    if labelFilename == None:
        nbrLignes = len(lignes)
        labels = [0]*nbrLignes # https://www.geeksforgeeks.org/python-which-is-faster-to-initialize-lists/
    else:
        labels = txtFileToListe(labelFilename)
    
    for i in range(len(labels)):
        lineToSVMfile(svmFilename, lignes[i], labels[i], index)


def LaunchSVM(nameEntrainementFile, namePrediction, nameTest=None, nameModel=None):
    fichierEntrainementLabel = "Normalisation/" + nameEntrainementFile + "label.txt"
    fichierEntrainementSVM = "GenerationFichier/SVM/" + nameEntrainementFile + ".svm"
    fichierEntrainement = "Normalisation/" + nameEntrainementFile + ".txt"
    fichierPrediction = "Normalisation/" + namePrediction + ".txt"
    fichierPredictionSVM = "GenerationFichier/SVM/" + namePrediction + ".svm"

    index = GetIndexFromNormaliseFile(fichierEntrainement)
    index = GetIndexFromNormaliseFile(fichierPrediction, index=index)
    print(fichierPrediction + " " + fichierEntrainement)
    
    TXTtoSVM(fichierEntrainement, fichierEntrainementSVM, index, labelFilename=fichierEntrainementLabel)
    TXTtoSVM(fichierPrediction, fichierPredictionSVM, index)

    fichierModel = ""
    if nameModel == None:
        fichierModel = "GenerationFichier/Models/SVM/tweetsDefault.model"
    else:
        fichierModel = "GenerationFichier/Models/SVM/" + nameModel + ".model"

    # Lancement de l'entrainement du model SVM :
    print("Lancement de l'entrainement du model SVM")
    subprocess.call(["liblinear-2.30/train","-c","1","-e","0.1",fichierEntrainementSVM,fichierModel])

    if(nameTest != None):    
        fichierTest = "Normalisation/" + nameTest + ".txt"
        fichierTestSVM = "GenerationFichier/SVM/" + nameTest + ".svm"
        fichierTestLabel = fichierTest.replace(".txt", "label.txt")
        index = GetIndexFromNormaliseFile(fichierTest, index)
        TXTtoSVM(fichierTest, fichierTestSVM, index, labelFilename=fichierTestLabel)

        # Lancement de la prediction sur les jeux de données testes :
        print("Lancement de la prediction sur les jeux de données testes")
        resultTestFile = "GenerationFichier/" + nameTest + "Result.txt"
        subprocess.call(["liblinear-2.30/predict",fichierTestSVM,fichierModel,resultTestFile])

    # Lancement de la prediction sur le sujet sans annotation : 
    print("Lancement de la prediction finale")
    resultPredictionFile = "GenerationFichier/" + namePrediction + ".txt"
    subprocess.call(["liblinear-2.30/predict",fichierPredictionSVM,fichierModel,resultPredictionFile])

    return resultPredictionFile

