
# VisionAI — Image Classifier

An AI image classification web application built with Python,
Streamlit, TensorFlow, and MobileNetV2.

## Features

- Upload images
- Preview uploaded images
- Predict the top three labels
- Display confidence percentages
- Show a confidence chart
- Validate image files and handle errors
- Explain model predictions

## Technologies

- Python
- Streamlit
- TensorFlow
- MobileNetV2
- Pillow
- NumPy
- Pandas

## How it works

1. The user uploads an image.
2. The application validates the file.
3. Pillow converts and resizes the image.
4. MobileNetV2 processes the image.
5. The model predicts ImageNet categories.
6. The application displays the top three labels
   and their confidence scores.

## Run locally

Install Python 3.11, then create a virtual environment.

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

Open the local URL shown in your terminal.

## Limitations

The pretrained model recognizes categories from ImageNet.
Its predictions are not guaranteed to be correct.
# VisionAI-Image-Classifier
