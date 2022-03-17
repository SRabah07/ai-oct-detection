# Description 
Le but du projet, est de mettre en place une IA, permettant de catégoriser les Retinal OCT Images (optical coherence tomography) en 4 classes : 
- Normal 
- CNV
- DME
- DRUSEN

J'ai opté pour la mise en place d'un classificateur par le biais de modèles de Deep Learning pour la classification d'images.

Merci de noter que le dataset est disponible sur [Kaggle](https://www.kaggle.com/paultimothymooney/kermany2018). J'en remercie les auteurs.


# Modélisation et Architecture des modèles
J'ai mis en place différentes architectures basées sur les réseaux de neurones convolutifs (CNN), le transfert learning ainsi que les modèles hybrides (combinant l'extraction de features avec les modèles de machine learning standard). Je les ai évalués avec les différentes métriques.

## Les modèles entraînés 

- Un modèle CNN personnalisé
- Le modèle LeNet
- Un modèle de Transfert Learning EfficientNetB5
- Un modèle de Transfer Learning VGG16

## Pré processing des données
Veuillez vous référer à la section dataviz du notebook `OCT.ipynb`.

## Modeling
Veuillez vous référer à la section Modeling du notebook `OCT.ipynb`. 

## Streamlit

J'ai mis en place une application web, en utilisant [Streamlit]( https://www.streamlit.io/). Cette application permettra de choisir un modèle, de télécharger un OCT et d'afficher le résultat de la classification. Voir [ici](https://github.com/SRabah07/ai-oct-detection/oct-streamlit.py) pour plus de détails sur l'implémentation.

L'utilisation de cette application, requiert l'exécution du notebook (voir ci-dessous), et de sauvegarder les modèles générés, dans le dossier **models**.

La commande qui lance l'application streamlit est : `streamlit run oct-streamlit.py`
