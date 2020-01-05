# -*- encoding: utf-8 -*-

from DocToVec import GetModelFromDocToVec
from gensim.utils import simple_preprocess
from projectHelper import txtFileToListe
from keras.layers import Dense, Dropout, Activation, Flatten, Input
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.callbacks import Callback
from keras.models import load_model
from projectHelper import LabelConverter
from projectHelper import WriteInFile
from keras import backend as K
from projectHelper import PredictionToOut
import numpy as np
from keras.callbacks import Callback
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score

# Initialisation des données standards
listesFichiersEntrainement = ["Normalisation/JSONdata.txt"]
tailleVecteurDocToVec = 100
pcTrain = 90
epochsDocToVec = 40
testingDocToVec = False
BestModelFile = ""

class Metrics(Callback):
    def on_train_begin(self, logs={}):
        global BestModelFile
        self.bestModelFile = BestModelFile
        self.bestModel = None
        self.FMesureMax = float(0)
        self.FMesureMaxDetail = []

    def on_epoch_start(self, epoch, logs={}):
        if not (self.bestModel == None):
            self.model = self.bestModel
    
    def on_epoch_end(self, epoch, logs={}):
        val_predict = (np.asarray(self.model.predict(self.validation_data[0]))).round()
        val_targ = self.validation_data[1]
        FMesureParLabel = []
        for label in range(0,4):
            nbrDocAppartenantLabel = 0
            nbrDocAttribuerLabel = 0
            nbrDocCorrectementAttribuerLabel = 0
            for i in range(0, len(val_targ)):
                if val_targ[i][label] == 1:
                    nbrDocAppartenantLabel += 1
                    if val_predict[i][label] == 1:
                        nbrDocCorrectementAttribuerLabel += 1
                else:
                    if val_predict[i][label] == 1:
                        nbrDocAttribuerLabel += 1
            nbrDocAttribuerLabel += nbrDocCorrectementAttribuerLabel
            try:
                precision = float(nbrDocCorrectementAttribuerLabel)/float(nbrDocAttribuerLabel)
            except ZeroDivisionError:
                precision = 0
            try:
                rappel = float(nbrDocCorrectementAttribuerLabel)/float(nbrDocAppartenantLabel)
            except ZeroDivisionError:
                rappel = 0
            Fmesure = 2*((precision*rappel)/(precision+rappel+K.epsilon()))
            FMesureParLabel.append(Fmesure)
        FMesureTotal = float(FMesureParLabel[0] + FMesureParLabel[1] + FMesureParLabel[2] + FMesureParLabel[3]) / 4
        #print("FMesure total = " + str(FMesureTotal) + "([autre]=" + str(FMesureParLabel[0]) + " [mixte]=" + str(FMesureParLabel[1]) +" [positif]=" + str(FMesureParLabel[2]) + " [negatif]=" + str(FMesureParLabel[3]) + ")")
        if FMesureTotal > self.FMesureMax:
            self.FMesureMax = FMesureTotal
            self.FMesureMaxDetail = FMesureParLabel
            self.model.save(self.bestModelFile)
            self.bestModel = self.model
        return

    def on_train_end(self, logs={}):
        print("Entrainement fini, Resultat : ")
        print("FMesure total = " + str(self.FMesureMax) + "([autre]=" + str(self.FMesureMaxDetail[0]) + " [mixte]=" + str(self.FMesureMaxDetail[1]) +" [positif]=" + str(self.FMesureMaxDetail[2]) + " [negatif]=" + str(self.FMesureMaxDetail[3]) + ")")
        


def InitDocToVecParameter(fichiersEntrainementDocToVec, tailleVecteur, epochs, pourcentageTrain, TestingDocToVec=False):
    global listesFichiersEntrainement, tailleVecteurDocToVec, pcTrain, epochsDocToVec, testingDocToVec

    listesFichiersEntrainement = []
    for fichier in fichiersEntrainementDocToVec:
        nomFichier = "Normalisation/" + fichier + ".txt"
        listesFichiersEntrainement.append(nomFichier)
    
    tailleVecteurDocToVec = tailleVecteur 
    epochsDocToVec = epochs
    pcTrain = pourcentageTrain
    testingDocToVec = TestingDocToVec

def vectorFromDataList(liste, docToVecModel):
    vectorData = []
    for data in liste:
        tokens = data.split(' ')
        vecteur = docToVecModel.infer_vector(tokens)
        vectorData.append(vecteur)
    return vectorData

def getFFWModel(shape, input, activationType, metrics):
    model = Sequential()
    first = True
    for layer in shape:
        if first:
            model.add(Dense(layer, activation=activationType, input_shape=(input,)))
            first = False
        else:
            model.add(Dense(layer, activation=activationType))
            model.add(Dropout(0.2))

    model.add(Dense(4, activation='softmax')) # Verifier Sigmoid :/
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=metrics)
    return model

def RNNscript(fichierEntrainement, predictFile, pourcentageTrain, shapeRNN=[200,150,100], nameModel="RNNmodel", typeRNN="FFW", TailleDuComitee=0, activationType='relu', batch_size=32, epochs=10):
    global listesFichiersEntrainement, tailleVecteurDocToVec, pcTrain, epochsDocToVec, testingDocToVec
    if typeRNN == "Committee" and TailleDuComitee == 0:
        print("Nombre d'expert non indiqué, veuillez renseigner le parametre TailleDuComitee")
        quit()
    docToVecModel = GetModelFromDocToVec(listesFichiersEntrainement, pcTrain, tailleVecteurDocToVec, epochsDocToVec, testing=testingDocToVec)
    docToVecModel.save("GenerationFichier/Models/DocToVec/" + nameModel + "_DocToVec")
    print("Lancement de la création du RNN")
    #except expression as identifier:
    fichierEntrainement = "Normalisation/" + fichierEntrainement + ".txt"
    fichierPredict = "Normalisation/" + predictFile + ".txt"
    fichierLabel = fichierEntrainement.replace(".txt", "label.txt")

    datas = txtFileToListe(fichierEntrainement, withSpaceTreatment=True)
    predictData = txtFileToListe(fichierPredict, withSpaceTreatment=True)
    labelDatas = txtFileToListe(fichierLabel, withSpaceTreatment=True)
    nbrTrain = int((float(len(datas)) / 100) * pourcentageTrain)

    trainData = datas[:nbrTrain]
    trainLabel = labelDatas[:nbrTrain]
    testData = datas[nbrTrain:]
    testLabel = labelDatas[nbrTrain:]

    vectorTrain = [vectorFromDataList(trainData, docToVecModel)]
    vectorToPredict = [vectorFromDataList(predictData, docToVecModel)]
    vectorTest = [vectorFromDataList(testData, docToVecModel)]

    trainLabel = np_utils.to_categorical(trainLabel, 4)
    testLabel = np_utils.to_categorical(testLabel, 4)

    print("Nombre de données d'entrainement : " + str(len(vectorTrain[0])))
    print("Nombre de données de teste : " + str(len(vectorTest[0])))
    metrics = ['accuracy'] 
    global BestModelFile

    if typeRNN == "FFW":
        print("Creation du model FFW")
        model = getFFWModel(shapeRNN, tailleVecteurDocToVec, activationType, metrics)
        BestModelFile = "GenerationFichier/Models/RNN/" + nameModel + ".hdf5"
        print("Entrainement du model")
        # Lancement de l'entrainement
        callbackMetric = Metrics()
        model.fit(vectorTrain, trainLabel, batch_size=batch_size, epochs=epochs, verbose=1, callbacks=[callbackMetric] ,validation_data=(vectorTest, testLabel))
        model = load_model(BestModelFile)
        score = model.evaluate(vectorTest, testLabel, verbose=0)
        print('Test loss:', score[0])
        print('Test accuracy:', score[1])
        results = model.predict(vectorToPredict)
        labelResult = []
        for result in results:
            listResult = result.tolist()
            indexMax = listResult.index(max(listResult))
            labelResult.append(str(indexMax) + "\n")
        WriteInFile("GenerationFichier/RNNtestResult.txt", labelResult)

    elif typeRNN =="Committee":
        models = []
        print("Creation du commitée (model FFW)")
        for i in range(0,TailleDuComitee):
            model = getFFWModel(shapeRNN, tailleVecteurDocToVec, activationType, metrics)
            BestModelFile = "GenerationFichier/Models/RNN/" + nameModel + str(i) + ".hdf5"
            callbackMetric = Metrics()
            model.fit(vectorTrain, trainLabel, batch_size=batch_size, epochs=epochs, verbose=0, callbacks=[callbackMetric] ,validation_data=(vectorTest, testLabel))
            models.append(load_model(BestModelFile))
            print(str((i+1)*(100/TailleDuComitee)) + "%")
        expertsResult = []
        for numExpert in range(0,TailleDuComitee):
            score = models[numExpert].evaluate(vectorTest, testLabel, verbose=0)
            poids = score[1]
            resultExpert = models[numExpert].predict(vectorToPredict).tolist()
            for y in range(0,len(resultExpert)):
                if len(expertsResult) == y:
                    expertsResult.append([i*poids for i in resultExpert[y]])
                else: 
                    for z in range(0, len(expertsResult[y])):
                        expertsResult[y][z] += (resultExpert[y][z] * poids)
        labelResult = []
        for result in expertsResult:
            indexMax = result.index(max(result))
            labelResult.append(str(indexMax) + "\n")
        WriteInFile("GenerationFichier/RNNtestResult.txt", labelResult)


        
            
            


    else:
        raise ValueError("Le model n'est pas reconnu : vérifié la valeur du type de réseau neuronal demandé")
