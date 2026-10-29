
import io

import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image, UnidentifiedImageError
from keras.applications.mobilenet_v2 import (
    MobileNetV2,
    preprocess_input,
    decode_predictions,
)

st.set_page_config(
    page_title="VisionAI - Image Classifier",
    page_icon="🔎",
    layout="wide",
)

MAX_FILE_SIZE_MB = 10
ALLOWED_FORMATS = {"JPEG", "PNG", "WEBP", "BMP"}

st.markdown("""
<style>
.stApp {
    background: radial-gradient(
        circle at top left,
        #162b4d 0%,
        #0b1020 45%,
        #070a12 100%
    );
    color: #f4f7ff;
}
[data-testid="stHeader"] {
    background: transparent;
}
.hero {
    padding: 25px;
    border: 1px solid #425a85;
    border-radius: 20px;
    background: linear-gradient(
        135deg,
        rgba(42, 83, 150, 0.45),
        rgba(93, 49, 130, 0.3)
    );
    margin-bottom: 20px;
}
</style>
""", unsafe_allow_html=True)


@st.cache_resource(show_spinner="Loading MobileNetV2 model...")
def load_model():
    return MobileNetV2(
        weights="imagenet",
        include_top=True
    )


def predict_image(image, model):
    image = image.convert("RGB").resize((224, 224))

    image_array = np.asarray(image, dtype=np.float32)
    image_array = np.expand_dims(image_array, axis=0)
    image_array = preprocess_input(image_array)

    predictions = model.predict(image_array, verbose=0)

    return decode_predictions(predictions, top=3)[0]


st.markdown("""
<div class="hero">
    <h1>🧖🏻‍♂️ VisionAI</h1>
    <h3>AI-Powered Image Classifier</h3>
    <p>Upload an image and let artificial intelligence analyze it.</p>
</div>
""", unsafe_allow_html=True)

st.write(
    "VisionAI uses TensorFlow and MobileNetV2 to identify "
    "possible objects in your image."
)

with st.sidebar:
    st.title("About VisionAI")
    st.write("**Language:** Python")
    st.write("**Interface:** Streamlit")
    st.write("**AI Framework:** TensorFlow")
    st.write("**Model:** MobileNetV2")
    st.write("**Classes:** ImageNet (1000 categories)")

    st.divider()

    st.caption(
        "Predictions are estimates and may be incorrect. "
        "The model recognizes categories from ImageNet."
    )

uploaded_file = st.file_uploader(
    "Upload an image",
    type=["jpg", "jpeg", "png", "webp", "bmp"],
    help="Maximum file size: 20 MB"
)

if uploaded_file is None:
    st.info("Upload an image to start classification.")
    st.stop()

if uploaded_file.size > MAX_FILE_SIZE_MB * 1024 * 1024:
    st.error("Image is too large. Please upload an image under 20 MB.")
    st.stop()

try:
    raw_image = uploaded_file.getvalue()

    with Image.open(io.BytesIO(raw_image)) as check:
        image_format = (check.format or "").upper()
        check.verify()

    if image_format not in ALLOWED_FORMATS:
        st.error("Unsupported image format.")
        st.stop()

    with Image.open(io.BytesIO(raw_image)) as source:
        image = source.convert("RGB")

except (UnidentifiedImageError, OSError, ValueError):
    st.error(
        "Invalid image file. Please upload a valid JPG, PNG, "
        "WEBP, or BMP image."
    )
    st.stop()

image_col, result_col = st.columns([1, 1.2], gap="large")

with image_col:
    st.subheader("Uploaded Image")
    st.image(
        image,
        caption=uploaded_file.name,
        use_container_width=True
    )
    st.caption(f"Dimensions: {image.width} × {image.height} pixels")

with result_col:
    st.subheader("Top 3 Predictions")

    try:
        model = load_model()

        with st.spinner("Analyzing your image..."):
            predictions = predict_image(image, model)

        results = []

        for rank, (_, label, score) in enumerate(
            predictions, start=1
        ):
            results.append({
                "Rank": rank,
                "Predicted Label": label.replace("_", " ").title(),
                "Confidence (%)": round(float(score) * 100, 2),
            })

        results_df = pd.DataFrame(results)
        best = results_df.iloc[0]

        metric1, metric2 = st.columns(2)

        metric1.metric(
            "Best Prediction",
            best["Predicted Label"]
        )

        metric2.metric(
            "Confidence",
            f'{best["Confidence (%)"]:.2f}%'
        )

        st.write("### Confidence Chart")

        chart_data = results_df.set_index(
            "Predicted Label"
        )[["Confidence (%)"]]

        st.bar_chart(
            chart_data,
            horizontal=True
        )

        st.write("### Prediction Details")

        st.dataframe(
            results_df,
            hide_index=True,
            use_container_width=True
        )

        st.write("### What does this mean?")

        st.write(
            f'The model ranked **{best["Predicted Label"]}** '
            f'first with a confidence score of '
            f'**{best["Confidence (%)"]:.2f}%**.'
        )

        st.info(
            "MobileNetV2 compares the image with learned ImageNet "
            "categories. The top three results are the categories "
            "with the highest model scores. A high score does not "
            "guarantee that the label is correct."
        )

    except Exception as error:
        st.error("Unable to classify this image.")

        st.write(
            "Check your internet connection, installed packages, "
            "and terminal logs. The first run downloads pretrained "
            "model weights."
        )

        with st.expander("Technical error"):
            st.code(str(error))

st.divider()

st.caption(
    "VisionAI | Python + Streamlit + TensorFlow + MobileNetV2"
)
