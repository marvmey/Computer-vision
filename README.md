# Computer-vision

Ce repository regroupe le résultat de mon travail effectué lors de mon stage chez Moviin


## Pour commencer

Ma tâche etait d'effectuer une reproduction en python du code du programme d'Halcon, utilisé en industrie et basé sur le computer-vision.
Le programme propose plusieurs approches :
- La classification
- La segmentation
- L'anomaly detection


### Les pré-requis : 

- Python 3.2 -> Python 3.10
- Pycharm
- Cuda, Cudnn (si vous possédez un GPU)
- Tensorflow

#### dataset
Vous pouvez retrouver un large choix de jeu de données sur le site :*https://www.mvtec.com/company/research/datasets/mvtec-ad*



## Classification

### Modèle

Utilisation d'un réseau pré entrainé et suppression de la dernière couche afin d'y ajouter la notre et faire correspondre le nombre de classe.
J'ai utilisé VGG16 et InceptionV3 avec Tensorflow. Voici les résultats:

| Model | Data | Acc | Loss | batch_size | shape | epoch |
| ---   |---    | --- | --- |---         | ---   | ---   |
| VGG16 | pills | 0.987| 0.15| 32 | 256,256,3 | 25|






