# -*- encoding: utf-8 -*-
import unidecode
import emoji
import re
import json

from os import listdir
from os.path import isfile, join
from lxml import etree
from IndexBuilder import IndexBuilder
from projectHelper import txtFileToListe
from projectHelper import LabelConverter
from projectHelper import WriteInFile
from projectHelper import ISO8859Converter

accents = {'a': ['à', 'ã', 'á', 'â','À', 'Ã', 'Á', 'Â'],
           'e': ['é', 'è', 'ê', 'ë','É', 'È', 'Ê', 'Ë'],
           'i': ['î', 'ï', 'Î', 'Î', 'Ï', 'Î'],
           'u': ['ù', 'ü', 'û', 'Ù', 'Ü', 'Û'],
           'o': ['ô', 'ö', 'Ô', 'Ö']}


MotsGrossiers = {}
ListeMotVide = {}

def ChargementMotsVide():
    # Récupéré tous les fichiers d'un dossier :
    # https://stackoverflow.com/questions/3207219/how-do-i-list-all-files-of-a-directory
    chemin = "Data/StopWord/"
    listeFichiers = [f for f in listdir(chemin) if isfile(join(chemin, f))]
    global ListeMotVide
    for fichier in listeFichiers:
        path = join(chemin, fichier)

        listeMotVide = txtFileToListe(path)

        for mot in listeMotVide:
            MVnormaliser = normalisationSimple(mot)
            if len(MVnormaliser.split()) == 1:
                ListeMotVide[MVnormaliser] = []

def ChargementMotsGrossier():
    listeMotsGrossier = txtFileToListe("Data/MotsGrossiers/MGFromWikipedia.txt")
    listeMotsGrossier += txtFileToListe("Data/MotsGrossiers/MGSupplementaire.txt")
    
    global MotsGrossiers
    for motGrossier in listeMotsGrossier:    
        motGrossierNormaliser = normalisationSimple(motGrossier)
        if len(motGrossierNormaliser.split()) == 1:
            MotsGrossiers[motGrossierNormaliser] = []
    listeMGnormaliser = []
    for motGrossier in MotsGrossiers.keys():
        listeMGnormaliser.append(motGrossier + "\n")
    WriteInFile("Normalisation/MG.txt", listeMGnormaliser)
    
def enleverAccent(message):
    global accents
    for lettre, accentPossibles in accents.iteritems():
        for accent in accentPossibles:
            if accent in message:
                message = message.replace(accent, lettre)
    return message


def Normalise_DataDeft2017(TypeNormalisation="Simple"):

    fileName = "Data/data_deft2017/task1-train.csv"
    print("Lancement de la normalisation de " + fileName)
    lignes = []
    labels = []
    with open(fileName) as file:
        for line in file:
            ligne = ""
            data = line.split('\t')
            if(len(data) > 1):
                message = normalisation(data[1], TypeNormalisation=TypeNormalisation)
                labelStr = data[2].replace('\n', '')
                label = str(LabelConverter(labelStr.replace('\r', '')))+ "\n"
                mots = message.split()
                for mot in mots:
                    ligne += mot + " "
                lignes.append(ligne + "\n")
                labels.append(label)
    WriteInFile("Normalisation/SVMtrain.txt", lignes)
    WriteInFile("Normalisation/SVMtrainlabel.txt", labels)

    lignes = []
    labels = []
    fileName = "Data/data_deft2017/task1-testGold.csv"
    print("Lancement de la normalisation de " + fileName)
    with open(fileName) as file:
        for line in file:
            ligne = ""
            data = line.split('\t')
            if(len(data) > 1):
                message = normalisation(data[1])
                labelStr = data[2].replace('\n', '')
                label = str(LabelConverter(labelStr.replace('\r', '')))+ "\n"
                mots = message.split()
                for mot in mots:
                    ligne += mot + " "
                lignes.append(ligne + "\n")
                labels.append(label)
    WriteInFile("Normalisation/SVMtest.txt", lignes)
    WriteInFile("Normalisation/SVMtestlabel.txt", labels)
    
def Normalise_PolariteMots():
    fileName = "Data/06032019-POLARITY-JEUXDEMOTS-FR.txt"
    print("Lancement de la normalisation de " + fileName)
    listePolarite = []
    nbrLigne10pc = int(1113399/10)
    nbrLigne = 0
    with open(fileName) as file:
        for line in file:
            if int(nbrLigne % nbrLigne10pc) == 0:
                print(str(nbrLigne/nbrLigne10pc) + "0% Effectué")
            nbrLigne += 1
            if not "//" in line:
                if line != "\n":
                    data = line.split('"')
                    if(len(data) != 3):
                        print(data)
                    text = normalisationSimple(ISO8859Converter(data[1]))
                    if(len(text.split()) == 1):
                        values = text.replace(' ', '')
                        polValue = data[2]
                        values += polValue
                        listePolarite.append(values)
    WriteInFile("Normalisation/Polarite.txt", listePolarite)


def Normalise_Unlabeled(TypeNormalisation="Simple"):

    fileName = "Data/unlabeled.xml"
    print("Lancement de la normalisation de " + fileName)
    tree = etree.parse(fileName)
    root = tree.xpath("/root")[0]
    lignesDic = {}
    nbTweet = 0
    for tweet in root.getchildren():
        if nbTweet % 7200 == 0:
            print(str((nbTweet/7200)+1)+ "0% Effectuer")
        nbTweet += 1
        elemMessage = tweet.find("message")
        message = normalisation(elemMessage.text, TypeNormalisation)
        message = message + "\n"
        if not lignesDic.has_key(message):
            lignesDic[message] = 0
    WriteInFile("Normalisation/70kTweet.txt", lignesDic)


def Normalise_Json(TypeNormalisation="Simple"):
    data = None
    print("Lancement de la normalisation de Data/tweetsAnnotate.json")
    with open("Data/tweetsAnnotate.json") as file:
        data = json.load(file)
    tweets = data["tweets"]
    listeTweet = []
    listeTweetlabel = []
    for tweet in tweets:
        listeTweet.append(normalisation(tweet["message"], TypeNormalisation=TypeNormalisation) + "\n")
        listeTweetlabel.append(str(LabelConverter(normalisationSimple(tweet["polarity"]))) + "\n")

    WriteInFile("Normalisation/JSONdata.txt", listeTweet)
    WriteInFile("Normalisation/JSONdatalabel.txt", listeTweetlabel)



def Normalise_DonneeTest(TypeNormalisation="Simple"):
    fileName = "Data/donneeTest.txt"
    print("Lancement de la normalisation de " + fileName)
    lignesSource = txtFileToListe(fileName)
    lignes = []
    ids = []
    for ligneSource in lignesSource:
        ligne = ""
        texte = ligneSource[19:].replace('\n', '')
        id = ligneSource[:18] + "\n"
        message = normalisation(texte, TypeNormalisation)
        mots = message.split()
        for mot in mots:
            ligne += mot + " "
        lignes.append(ligne + "\n")
        ids.append(id)
    WriteInFile("Normalisation/idsDT.txt", ids)
    WriteInFile("Normalisation/texteDT.txt", lignes)

def normalisation(message, TypeNormalisation="Simple"):
    result = normalisationSimple(message)

    if TypeNormalisation != "Simple":
        global MotsGrossiers, ListeMotVide
        mots = result.split()
        result = ""
        for mot in mots:
            # remplacer les mots grossiers par "Insulte" et retrait des mots vides
            if MotsGrossiers.has_key(mot):
                result += "Insulte "
            if not ListeMotVide.has_key(mot):
                result += mot + " "

    return result

def normalisationSimple(message):
    # Enlever les accents
    if type(message) == type("str"):
        message = enleverAccent(message)
        uni = unicode(message, 'utf-8')
        uni = emoji.demojize(uni)
        message = unidecode.unidecode(uni)
    else:
        uni = emoji.demojize(message)
        message = unidecode.unidecode(uni)
    # Transformation en minuscule
    message = message.lower()

    # Enlever les liens
    if "http" in message:
        mots = message.split(' ')
        for mot in mots:
            if "http" in mot:
                message = message.replace(mot, '')

    # On enleve les caracteres spéciaux
    stri = '&"{[(-|)]=}\'/*-!?+:;.,<>»«…'
    for s in stri:
        message = message.replace(s, ' ')
    # Si les mots contient plus de 3 fois la même lettre, on les remplacent par la premiere occurence (merciiiiiiiii = merci)
    multipleCarac = re.findall(
        "[a]{3,}|[b]{3,}|[c]{3,}|[d]{3,}|[e]{3,}|[f]{3,}|[g]{3,}|[h]{3,}|[i]{3,}|[j]{3,}|[k]{3,}|[l]{3,}|[m]{3,}|[n]{3,}|[o]{3,}|[p]{3,}|[q]{3,}|[r]{3,}|[s]{3,}|[t]{3,}|[u]{3,}|[v]{3,}|[w]{3,}|[x]{3,}|[y]{3,}|[z]{3,}", message)
    while len(multipleCarac) != 0:
        for carac in multipleCarac:
            message = message.replace(carac, str(carac[0]))
        multipleCarac = re.findall(
            "[a]{3,}|[b]{3,}|[c]{3,}|[d]{3,}|[e]{3,}|[f]{3,}|[g]{3,}|[h]{3,}|[i]{3,}|[j]{3,}|[k]{3,}|[l]{3,}|[m]{3,}|[n]{3,}|[o]{3,}|[p]{3,}|[q]{3,}|[r]{3,}|[s]{3,}|[t]{3,}|[u]{3,}|[v]{3,}|[w]{3,}|[x]{3,}|[y]{3,}|[z]{3,}", message)

    return message


def CorpusLabelWordToPolarity():
    ChargementMotsGrossier()
    ChargementMotsVide()
    Normalise_PolariteMots()
    Normalise_Unlabeled(TypeNormalisation="Complete")
    Normalise_DonneeTest(TypeNormalisation="Complete")

def JsonToSVMSimple():
    ChargementMotsGrossier()
    ChargementMotsVide()
    Normalise_Json()
    Normalise_DonneeTest()

#CorpusLabeled()
#ChargementMotsVide()
#Normalise_PolariteMots()
#LaunchNormalisation()
