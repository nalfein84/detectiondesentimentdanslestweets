# Détection de sentiment dans les tweets

## Introduction 

Ce projet est le résultat du défi de l'UE (Unité d'Enseignement) Application d'Innovation proposé au Centre d'Enseignement et de Recherche Informatique (CERI) d'Avignon. Il a étais mené dans son intégralité par Battis Yacine et moi, PASSEMARD Kévin.

### But du projet :

L’objectif du défi est de fournir une solution permettant de classifier automatiquement des
messages courts (ici des tweets) selon leur polarité (positif, négatif…) dans le cadre du débat de
l’entre-deux tours de l’élection présidentielle française 2017. 

Il s’organise autour de trois activités principales : 

1) la constitution d’un corpus 
2) la réalisation de l’outil permettant la détection de polarité
3) l’écriture d’un article présentant la solution retenue

Nous avons eu à notre disposition [un corpus de tweets non-annoté](Data/unlabeled.xml). 

### Liste des données utilisées :

Pour mener a bien notre projet nous avons utilisé plusieurs autres corpus de document lors de nos recherche :
* Un corpus déja annoté que le CERI nous a fournit lors d'un TP précédent comportant [un jeu de donnée de test](Data/data_deft2017/task1-testGold.csv) et [un jeu de donnée d'entrainement](Data/data_deft2017/task1-train.csv) (Taille Entrainement: 3906 Tweets, Taille test : 976 Tweets) 
* Un jeu de données comportant des polarités de mots annoté par une communauté sur le jeu [likeit](http://www.jeuxdemots.org/likeit.php) regroupant 1.113.390 mots (avec doublons) ainsi que leurs polaritées.  
* Un jeu de données que plusieurs groupe ont aidé a alimenté et crée initialement par le groupe xxx
* Une [liste de mots grossier](Data/MotsGrossiers) récupéré [sur wikipedia](https://fr.wiktionary.org/wiki/Cat%C3%A9gorie:Insultes_en_fran%C3%A7ais) et donc certain mots ont était [rajouté par nos soins](Data/MotsGrossiers/MGSupplementaire.txt)

## Dépendances

Pour pouvoir faire fonctionné l'integralité du projet il faut les librairies Python ci-dessous :
* unidecode
* emoji
* re
* json
* lxml
* gensim
* tensorflow
* keras
* numpy

Ainsi que la bibliotheque liblinear, fournit dans ce projet, permettant de crée et d'entrainer la SVM.

## Explication des fichiers .py

Le projets ce découpe en plusieurs fichiers python :
* Le fichier [projectHelper.py](projectHelper.py) permet de mettre a disposition des autres scriptes, des méthodes simples pour accéder aux fichiers du projet ainsi que certaine transformation banale (concatenation/transformation de label/encodage etc...)
* Le fichier [Normalisation.py](Normalisation.py) permet la normalisation de toutes les données nécessaires au projet, il crée les fichiers et les places dans un dossier nomée **Normalisation** qui dois être présent a la racine du projet.
* Le fichier [TweetLabeliserByPolarity.py](TweetLabeliserByPolarity.py) permet de constituer un corpus automatiquement grace à la polarisation des mots. Néanmoins cette approche n'est pas très concluante du a un fort biait psychologique présents sur l'annotation en lui même et est par concéquant fortement déprécié. 
* Le fichier [SVMscript.py](SVMscript.py) permet le lancement complet d'un algorithme SVM a partir de fichiers déja normaliser
* Le fichier [RNNscript.py](RNNscript.py) permet le lancement complet d'un réseau neuronal automatique a partir de fichiers normaliser 

## Remerciement 

Nous remercions les enseignants-chercheurs Richard Dufour, Vincent Labatut et Mickaël Rouvier responsable de l'UE, ainsi que tous les membres du CERI.
