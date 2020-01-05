# -*- encoding: utf-8 -*-

import re
from IndexBuilder import IndexBuilder

def ConcatFile(filepath1, filepath2, filepathResult, withLabel=False):
    lignesFich1 = txtFileToListe(filepath1, withEndLigne=True)
    lignesFich2 = txtFileToListe(filepath2, withEndLigne=True)

    lignesAll = lignesFich1 + lignesFich2
    WriteInFile(filepathResult, lignesAll)

    if withLabel:
        labelfile1 = filepath1.replace(".txt", "label.txt")
        labelfile2 = filepath2.replace(".txt", "label.txt")
        filepathResultLabel = filepathResult.replace(".txt", "label.txt")
        lignesLabel1 = txtFileToListe(labelfile1, withEndLigne=True)
        lignesLabel2 = txtFileToListe(labelfile2, withEndLigne=True)
        lignesAllLabel = lignesLabel1 + lignesLabel2
        WriteInFile(filepathResultLabel, lignesAllLabel)

def DeleteMultipleSpaceFromSentence(sentence):
    result = ""
    for word in sentence.split(' '):
        if word != "":
            result += word + " "
    
    result = result[:len(result)-1]
    return result

def GetIndexFromNormaliseFile(filename,index=None):
    if index == None:
        index = IndexBuilder()
    print("Creation de l'index pour le fichier : " + filename)

    lignes = txtFileToListe(filename)
    for ligne in lignes:
        index.AddElem(ligne)
    return index

def PredictionToOut(resultPredictionFileName):
    print("Starting ...")
    out = txtFileToListe(resultPredictionFileName)
    data = txtFileToListe("Normalisation/idsDT.txt")
    print("Format des inputs : " + str(len(data)) + " x " + str(len(out)))
    lignes = []
    for i in range(len(out)):
        idTweet = data[i]
        label = LabelConverter(int(out[i]))
        ligneRes = idTweet + " " + label + "\n"
        lignes.append(ligneRes)
    WriteInFile('GenerationFichier/ResultatFinal.txt', lignes)
    print("... and done")

def getIndexFromTP(index=None):
    if index == None:
        index = IndexBuilder()

    fileName = "Normalisation/SVMtrain.txt"
    index = GetIndexFromNormaliseFile(fileName, index)
    fileName = "Normalisation/SVMtest.txt"
    index = GetIndexFromNormaliseFile(fileName, index)
    
    return index

def GetIDsFromDonneeTest():
    return txtFileToListe("Normalisation/idsDT.txt")

def getIndexFromDonneeTest(index=None):
    if index == None:
        index = IndexBuilder()

    fileName = "Normalisation/texteDT.txt"
    index = GetIndexFromNormaliseFile(fileName, index)

    return index


def getIndexFromTweet(index=None):
    if index == None:
        index = IndexBuilder()

    fileName = "Normalisation/70kTweet.txt"
    index = GetIndexFromNormaliseFile(fileName, index)

    return index

def ISO8859Converter(string):
    return string.decode('iso-8859-1')

def MakeFileWithUniqueValue(filename, ReplaceFile=True):
    listData = txtFileToListe(filename, withEndLigne=True)
    unique = {}
    for data in listData:
        if not unique.has_key(data):
            unique[data] = 0

    resultFile = filename
    if not ReplaceFile:
        resultFile = filename.replace(".txt", "_unique.txt")
    WriteInFile(resultFile, unique.keys())


def LabelConverter(label):
    if type(label) == type("str"):
        if label == "objective" or label == "neutral":
            return 0
        elif label == "mixed":
            return 1
        elif label == "positive":
            return 2
        elif label == "negative":
            return 3
    else:
        if label == 0:
            return "autre"
        elif label == 1:
            return "mixte"
        elif label == 2:
            return "positif"
        elif label == 3:
            return "negatif"

def txtFileToListe(path, withEndLigne=False, withSpaceTreatment=False):
    liste = []
    file = open(path)
    lines = file.readlines()
    for line in lines:
        if not withEndLigne:
            line = line.replace('\n', '')
        line = line.replace('\r', '')
        if withSpaceTreatment:
            line = DeleteMultipleSpaceFromSentence(line)
        liste.append(line)
        
    file.close()
    return liste

def WriteInFile(filename, lignes):
    with open(filename, 'w') as fn:
        fn.writelines(lignes)

def matriceCorrelationToDictionnary():
    dictionnary = {}
    file = open("matriceCorrelation.txt")
    lines = file.readlines()
    for line in lines:
        ligneDouble = line.split(':')
        dictionnary[ligneDouble[0]] = []
        stringWithValues = ligneDouble[1]
        stringForReplace = "[]\n"
        for car in stringForReplace:
            stringWithValues = stringWithValues.replace(car, '')
        values = stringWithValues.split(', ')
        for value in values:
            dictionnary[ligneDouble[0]].append(value)

    file.close()
    return dictionnary
