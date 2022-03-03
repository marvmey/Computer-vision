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
Vous pouvez retrouver un large choix de jeu de données sur le site *https://www.mvtec.com/company/research/datasets/mvtec-ad*



## Classification

### Modèle

Utilisation d'un réseau pré entrainé et suppression de la dernière couche afin d'y ajouter la notre et faire correspondre le nombre de classe.
J'ai utilisé VGG16 et InceptionV3 avec Tensorflow. Voici les résultats sur le dataset des pillules :

| Model | Acc  | Loss | batch_size | shape     | epoch | optimizer|
| ---   | ---  | --- |---          | ---       | ---   | ---      |
| VGG16 | 0.987| 0.15| 32          | 256,256,3 | 25    |  Adam |
| VGG16 | 0.977| 0.49| 32          | 300,300,3 | 32    | Adam |
| VGG16 | 0.992| 0.1 | 64          | 300,300,3 | 150   | Adam |
| VGG16 | 0.989| 0.1 | 32          | 300,300,3 | 50    | Adam |
| InceptionV3| 0.991| 0.02| 32 |300,300,3| 50| Adam |
| InceptionV3| 0.986| 0.05| 32 | 300,300,3| 50| Adam  |    *** > Shuffle = True***
|inceptionV3| 0.985| 0.11| 32| 256,256,3 | 150| Adam|

### Variables pour * classifier.py * :


epochs =                    *nombre d'epoch*

dir_dataset = ''            *chemin du dataset input*

batch_size =                *taille du batch_size*

target_size = (,)           *taille de l'input lors du preprocessing*

input_shape= (,,)           *taille de l'input du modèle (= target_size, rgb or grayscale)*

shuffle = False             *melange du dataset lors du model.fit)*

optimizers =                *choix de l'optimizers ( Adam, RMS, SGD,...)*

name_of_model =''           *nom du modèle*

num_class =                 *nombre de classe à  prédire ( pills = 9)*

class_mode =                *binary, categorical, sparse or None*

color_mode =                *rgb or grayscale*

SEED =                      *graine pour la RNG*




### Gradcam

Implémentation de la fonction Gradcam permettant de mettre en couleur les zones ciblées par le modèle.
Exemple : 


<img src="https://github.com/marvmey/Computer-vision/blob/main/classifier/image_grad_cam/pill_ginseng_contamination_021.png" width="300" height="150">
<img src="https://github.com/marvmey/Computer-vision/blob/main/classifier/image_grad_cam/pill_ginseng_contamination_pill_ginseng_contamination_021.png" width="300" height="150">

<img src="https://github.com/marvmey/Computer-vision/blob/main/classifier/image_grad_cam/pill_ginseng_crack_022.png" width="300" height="150">
<img src="https://github.com/marvmey/Computer-vision/blob/main/classifier/image_grad_cam/pill_ginseng_crack_pill_ginseng_crack_022.png" width="300" height="150">


### Variables pour *Gradcam.py*  :


model_path = '' *chemin où le modèle a été sauvegardé*

dir_img_grad_cam =  *chemin du dossier où les images grad_cam seront sauvegardée* 

dir_dataset =  *chemin du dataset initial*

target_size = *taille de l'image en input*

img_size =  *taille de l'image souhaitée*

last_conv_layer_name = *dernière couche de convolution à récupérer*

prefix_to_remove = *si vous souhaitez retirer le préfix pour plus de vision dans le nom du fichier*











