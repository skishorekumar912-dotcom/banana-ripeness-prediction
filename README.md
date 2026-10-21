# Banana Ripeness & Spoilage Prediction

An educational demo that uses a trained MobileNetV2 image-classification model to classify banana images into four ripeness stages.

## Features

- Upload a banana image for classification.
- Predict one of four stages: unripe, ripe, overripe, or rotten.
- Display the model's output score.
- Show an estimated remaining shelf life using temperature and humidity inputs.
- Validate uploaded image formats and file size.

## Technology

- Python
- TensorFlow / Keras
- MobileNetV2
- Streamlit
- Pillow and NumPy

## Model

The model expects RGB images resized to 224 x 224 pixels and uses MobileNetV2 preprocessing.

The trained model file is stored in:

`Trained Models (.keras + .tflite)/banana_stage_mobilenet_final (1).keras`

The model output labels are ordered as follows:

1. `stage0_unripe`
2. `stage1_ripe`
3. `stage2_overripe`
4. `stage3_rotten`

## Run locally

Install the dependencies:

```bash
pip install -r requirements.txt
```

Start the Streamlit app:

```bash
streamlit run streamlit_app.py
```

## Important limitations

This is an educational demonstration. The model's output score is not necessarily a calibrated probability. The remaining-days figure is a rule-based estimate, not a separately validated shelf-life prediction or food-safety guarantee.

Model accuracy has not been independently verified as part of this deployment setup.

## Dataset and privacy

Training datasets and virtual-environment files are excluded from the repository. The trained model and application source code are included.

## Status

Public deployment and cloud inference still need to be tested.
