# Détection de sentiment dans les tweets

## Introduction 

Ce projet est le résultat du défi de l'UE (Unité d'Enseignement) Application d'Innovation proposé au Centre d'Enseignement et de Recherche Informatique (CERI) d'Avignon.

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

## Dépendances

Pour pouvoir faire fonctionner l'integralité du projet il faut les librairies Python ci-dessous :
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

## Remerciement 
