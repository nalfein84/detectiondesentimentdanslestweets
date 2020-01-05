# -*- encoding: utf-8 -*-

from gensim.utils import simple_preprocess
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from projectHelper import txtFileToListe
import random

# Exemple utilisé pour apprendre l'utilisation de la librairie Gensim pour le Doc2Vec :
# https://radimrehurek.com/gensim/auto_examples/tutorials/run_doc2vec_lee.html#sphx-glr-auto-examples-tutorials-run-doc2vec-lee-py 


def CorpusToDocAndToken(corpus, tokenOnly=False):
    result = []
    i = 0
    for line in corpus:
            tokens = line.split(" ")
            if tokenOnly:
                result.append(tokens)
            else:
                # For training data, add tags
                result.append(TaggedDocument(tokens, [i]))
                i +=1
    return result

def Testing(test_corpus, train_corpus, model):
    # Pick a random document from the test corpus and infer a vector from the model
    doc_id = random.randint(0, len(test_corpus) - 1)
    inferred_vector = model.infer_vector(test_corpus[doc_id])
    sims = model.docvecs.most_similar([inferred_vector], topn=len(model.docvecs))

    # Compare and print the most/median/least similar documents from the train corpus
    print('Test Document ({}): «{}»\n'.format(doc_id, ' '.join(test_corpus[doc_id])))
    for label, index in [('MOST', 0), ('MEDIAN', len(sims)//2), ('LEAST', len(sims) - 1)]:
        print(u'%s %s: «%s»\n' % (label, sims[index], ' '.join(train_corpus[sims[index][0]].words)))

def GetModelFromDocToVec(filenames, pcTrain, vector_size, epochs, testing=False):
    docs = []
    for filename in filenames:
        print("Lectures des données sur le fichier " + filename)
        docs += txtFileToListe(filename, withSpaceTreatment=True)
    nbrTrain = int((float(len(docs)) / 100) * pcTrain)

    trainData = CorpusToDocAndToken(docs[:nbrTrain])
    testData = CorpusToDocAndToken(docs[nbrTrain:], tokenOnly=True)
    
    
    print("Taille du corpus d'apprentissage : " + str(len(trainData)))
    print("Taille du corpus de teste : " + str(len(testData)))

    print("Apprentissage du model")
    model = Doc2Vec(vector_size=vector_size, min_count=1, epochs=epochs)
    model.build_vocab(trainData)
    model.train(trainData, total_examples=model.corpus_count, epochs=model.epochs)
    if testing:
        Testing(testData, trainData, model)
    return model
