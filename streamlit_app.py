import streamlit as st
import easyocr
import cv2
import numpy as np
from PIL import Image
import language_tool_python
import sqlite3
import io

# Initialisation OCR et correcteur
reader = easyocr.Reader(['fr'])
tool = language_tool_python.LanguageTool('fr-FR')

# Création/connexion DB
conn = sqlite3.connect('corrections.db')
c = conn.cursor()
c.execute('''CREATE TABLE IF NOT EXISTS corrections
             (original TEXT, corrected TEXT)''')
conn.commit()

# Fonction pour sauvegarder correction manuelle
def save_correction(original, corrected):
    c.execute("INSERT INTO corrections (original, corrected) VALUES (?, ?)", (original, corrected))
    conn.commit()

# Fonction pour vérifier si correction existe déjà
def get_saved_correction(original):
    c.execute("SELECT corrected FROM corrections WHERE original = ?", (original,))
    result = c.fetchone()
    return result[0] if result else None

st.title("Extraction et Correction de Texte OCR (Français)")
uploaded_file = st.file_uploader("Choisissez une image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file)
    image_np = np.array(image)

    # Correction de la rotation (optionnelle, ici basique)
    gray = cv2.cvtColor(image_np, cv2.COLOR_BGR2GRAY)
    coords = np.column_stack(np.where(gray > 0))
    angle = cv2.minAreaRect(coords)[-1]
    if angle < -45:
        angle = -(90 + angle)
    else:
        angle = -angle
    
    (h, w) = image_np.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(image_np, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)

    results = reader.readtext(rotated)

    st.image(rotated, caption="Image analysée", use_column_width=True)

    for i, (bbox, text, prob) in enumerate(results):
        col1, col2 = st.columns([1, 2])

        # Extraire portion image
        (tl, tr, br, bl) = bbox
        x_min = int(min(tl[0], bl[0]))
        y_min = int(min(tl[1], tr[1]))
        x_max = int(max(tr[0], br[0]))
        y_max = int(max(bl[1], br[1]))
        cropped = rotated[y_min:y_max, x_min:x_max]

        # Affichage
        with col1:
            st.image(cropped, width=150)

        with col2:
            text = text.strip()
            corrected = get_saved_correction(text) or tool.correct(text)
            user_input = st.text_input(f"Correction pour : '{text}'", value=corrected, key=f"corr_{i}")

            if user_input != corrected:
                if st.button(f"Sauvegarder correction #{i}", key=f"btn_{i}"):
                    save_correction(text, user_input)
                    st.success("Correction sauvegardée !")