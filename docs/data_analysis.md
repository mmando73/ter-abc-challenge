# Analyse exploratoire des données

## 1. Description du jeu de données
Les données fournies sont issues de la base ABC et divisées en deux sous-ensembles :
- **Train** : 198 modèles
- **Validation** : 50 modèles

Chaque modèle est défini par sa géométrie au format `.ply` (coordonnées x, y, z et normales nx, ny, nz principalement) et ses annotations de vérité terrain au format `.lb` (0 = non-arête, 1 = arête).

## 2. Volumétrie et distribution - Train dataset 
- **Points totaux :** 3 174 768 points
- **Moyenne par modèle :** 16 034 points
- **Modèle le plus dense :** `1452.ply`, avec 118 332 points
- **Modèle le moins dense :** `2340.ply`, avec 1 246 points

**Distribution des classes :**
- **Points arêtes (label 1) :** 149 469 points, soit 4.71%
- **Points non-arêtes (label 0) :** 3 025 299 points, soit 95.29%

## 3. Volumétrie et distribution - Validation dataset
- **Points totaux :** 690 211 points
- **Moyenne par modèle :** 13 804 points
- **Modèle le plus dense :** `0353.ply`, soit 34 340 points
- **Modèle le moins dense :** `0713.ply`, soit 1 004 points

**Distribution des classes :**
- **Points arêtes (label 1) :** 40 089, soit 5.81%
- **Points non-arêtes (label 0) :** 650 122; soit 94.19%

## 4. Analyse des extrêmes
L'analyse statistique par modèle montre des variations dans la densité des arêtes, liées à la forme spécifique des objets :

- **Densité minimale d'arêtes (dataset Train) :** `0310.ply` (0.61%)
    - ![train_0310](/docs/images/train_0310.png)

- **Densité maximale d'arêtes (dataset Train) :** `2800.ply` (38.52%)
    - ![train_2800](/docs/images/train_2800.png)

- **Densité minimale d'arêtes (dataset Validation) :** `0939.ply` (1.23%)
    - ![validation_0939](/docs/images/validation_0939.png)

- **Densité maximale d'arêtes (dataset Validation) :** `0713.ply` (87.65%)
    - *Note d'analyse :* Après vérification de la géométrie dans CloudCompare, ce taux extrême s'explique par la nature de l'objet (type fil de fer/rondelle/joint). Ce que nous intéresse ici ce qu'il ne s'agit pas d'une erreur d'annotation.
    - ![validation_0713](/docs/images/validation_0713.png)


