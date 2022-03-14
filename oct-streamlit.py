"""
- pip install opencv-python
- pip install streamlit
- pip install tensorflow
"""

from os import write
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 
import cv2 
import tensorflow as tf

# Encoding des différentes classes
ENCODING = {'CNV': 0, 'DME': 1, 'DRUSEN': 2, 'NORMAL': 3}
IMAGES_TYPES = list(ENCODING.keys())

st.title("Prédire la classe d'une Tomographie en Cohérence Optique")
st.subheader("Définition [Wikipedia] : ")

description="""
La tomographie en cohérence optique1 ou tomographie optique cohérente (TCO ou (en) OCT) est
une technique d'imagerie médicale bien établie qui utilise une onde lumineuse pour capturer des
images tridimensionnelles d'un matériau qui diffuse la lumière (par exemple un tissu biologique), 
avec une résolution de l'ordre du micromètre (1 µm).
La tomographie en cohérence optique est basée sur une technique interférométrique à faible cohérence, 
utilisant habituellement une lumière dans l'infrarouge proche.
"""

st.write('', description)

df = pd.DataFrame({
  'Nom du Modèle': ["CNN", "LeNet", "EfficientNetB5", "VGG16"],
  'Description': ["CNN", "LeNet", "EfficientNetB5", "VGG16"],
  'Chemin':['CNN', "LeNet", 'EfficientNetB5', 'VGG16']
 })


st.subheader("Veuillez choisir un modèle : ")
option = st.selectbox('', df['Nom du Modèle'])

'Vous avez selectionné le modèle : ', option

df_model = df[df['Nom du Modèle'] == option]
st.subheader("Architrcture et Métriques : ")

# Chemin vers l'architecture
chemin = f"./models/{df_model['Chemin'].values[0]}"

# Afficher le summary du modèle
model_summary = open(f"{chemin}/summary.txt").read()
st.text(model_summary)

# Afficher la loss accuracy
loss_image = plt.imread(f"{chemin}/loss_accuracy.png")
st.image(loss_image) 

# Afficher la matrice de confusion
confusion_image = plt.imread(f"{chemin}/confusion_matrix.png")
st.image(confusion_image) 

# Afficher le rapport de classification
classification_report = open(f"{chemin}/classification_report.txt").read()
st.text(classification_report)


# Load le modèle Keras stocké
uploaded_file = st.file_uploader("Télécharger votre Tomographie", type=['png','jpeg', 'jpg'])
if uploaded_file is not None:
    st.write(uploaded_file)
    image = plt.imread(uploaded_file)
    image = cv2.resize(image, dsize = (256, 256))
    st.image(image)

    if option not in "VGG16":
        # input is (256, 256)and we should convert (256, 256, 1)
        image=image.reshape(256, 256, 1)
    else:
        # VGG16 expects (256, 256, 3) 
        image=cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        st.write(image.shape)
    # Mettre l'image dans un numpy array.
    array=np.array(([image]))

    st.write(f"Loading model {option}...")
    model = tf.keras.models.load_model(f"{chemin}")
    st.write("Model loaded.")
    st.write("Start Prediction...")
    classes = model.predict(array)
    st.write("Prediction done.")
    # Récupérer l'index de la classe
    index = np.argmax(classes[0], axis=0)
    # calculer le pourcentage
    valeur = classes[0][index]
    valeur = round(valeur*100, 2)
    df_prediction = pd.DataFrame(data=classes[0], index = IMAGES_TYPES)
    df_prediction = df_prediction.rename({'0': 'Prédiction'})
    st.write(df_prediction)
    for key in ENCODING.keys():
        if index == ENCODING[key]:
            type_oct=key
    st.write(f'OCT de type {type_oct} à {valeur}%')
    if type_oct in uploaded_file.name:
        st.image('./icons/check.png', width=None)
    else:
        # i.e => 'NORMAL-5324912-1.jpeg'
        real_value = uploaded_file.name.split('-')[0]
        st.error(f'Should be {real_value}')
        st.image('./icons/cross.png', width=None)