import streamlit as st
import tensorflow as tf
from tensorflow.keras.preprocessing.image import img_to_array
import numpy as np
from PIL import Image
import requests
from io import BytesIO

# -------------------------------
# APP SETTINGS
# -------------------------------
st.set_page_config(page_title="🐶🐱 Dogs vs Cats Classifier", page_icon="🐾", layout="centered")

st.markdown(
    """
    <h1 style='text-align:center;'>🐶🐱 Dogs vs Cats Classifier</h1>
    <p style='text-align:center; font-size:18px; color:gray;'>
        Upload an image or paste a URL to classify it as a Dog or Cat using a Custom CNN model.
    </p>
    """,
    unsafe_allow_html=True
)

st.markdown("---")

# -------------------------------
# MODEL LOADING
# -------------------------------
@st.cache_resource
def load_cnn_model(path):
    return tf.keras.models.load_model(path)

model = load_cnn_model("best_model.best.h5")
img_size = (150, 150)

# -------------------------------
# IMAGE INPUT SECTION
# -------------------------------
st.subheader("📸 Upload or Paste Image URL")

col1, col2 = st.columns([1, 1])

with col1:
    uploaded_file = st.file_uploader("Upload Image", type=["jpg", "jpeg", "png"])

with col2:
    url_input = st.text_input("...or paste an image URL here")

image = None

if uploaded_file is not None:
    image = Image.open(uploaded_file)
elif url_input:
    try:
        response = requests.get(url_input)
        image = Image.open(BytesIO(response.content))
    except:
        st.error("❌ Could not load image from URL.")

# -------------------------------
# IMAGE DISPLAY & PREDICTION
# -------------------------------
if image is not None:
    st.markdown("### 🖼️ Image Preview")
    st.image(image, caption="Input Image", use_container_width=True)

    st.markdown("")

    if st.button("🚀 Predict", use_container_width=True):
        with st.spinner("Analyzing image... Please wait ⏳"):
            img = image.resize(img_size)
            img_array = img_to_array(img)
            img_array = np.expand_dims(img_array, axis=0)
            img_array = img_array / 255.0

            pred = model.predict(img_array)[0][0]
            label = "🐶 Dog" if pred >= 0.5 else "🐱 Cat"
            confidence = pred if pred >= 0.5 else 1 - pred

        st.success(f"### ✅ Prediction: {label}")
        st.write(f"**Confidence:** {confidence:.2%}")

else:
    st.info("👆 Upload an image or paste a URL above to get started.")

# -------------------------------
# FOOTER
# -------------------------------
st.markdown("---")
st.markdown(
    """
    <div style='text-align:center; color:gray;'>
        Created by <b>Mohamed Abobakr</b> 🧠 | Powered by <b>Streamlit</b> & <b>TensorFlow</b>
    </div>
    """,
    unsafe_allow_html=True
)
