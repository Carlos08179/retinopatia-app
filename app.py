import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from PIL import Image
import requests
import os

# Descargar el modelo desde Dropbox si no está presente
model_path = "binary_classifier_final.h5"
if not os.path.exists(model_path):
    url = "https://www.dropbox.com/scl/fi/akxu070jfv0og7b9pmks7/binary_classifier_final.h5?rlkey=d05akpuon45os716lcg9cral9&st=z9nkaq9b&dl=1"
    with requests.get(url, stream=True) as r:
        r.raise_for_status()
        with open(model_path, "wb") as f:
            for chunk in r.iter_content(chunk_size=8192):
                f.write(chunk)

# Cargar el modelo
model = load_model(model_path)

# Interfaz
st.title("Detector de Retinopatía")
st.write("Sube una imagen de ojo para analizar si hay retinopatía.")

# Subir imagen
uploaded_file = st.file_uploader("Selecciona una imagen", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    img = Image.open(uploaded_file).convert("RGB")
    st.image(img, caption="Imagen cargada", use_column_width=True)

    # Preprocesar imagen
    img = img.resize((224, 224))  # Ajusta al tamaño usado en tu modelo
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Predicción
    pred = model.predict(img_array)
    resultado = "Con retinopatía" if pred[0][0] > 0.5 else "Sin retinopatía"

    st.subheader(f"Resultado: {resultado}")




