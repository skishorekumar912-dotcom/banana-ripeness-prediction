import io
import math
import warnings
from pathlib import Path

import numpy as np
import streamlit as st
import tensorflow as tf
from PIL import Image, UnidentifiedImageError
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.utils import img_to_array, load_img


ROOT = Path(__file__).resolve().parent
MODEL_PATH = (
    ROOT
    / "Trained Models (.keras + .tflite)"
    / "banana_stage_mobilenet_final (1).keras"
)

IMG_SIZE = 224
MAX_UPLOAD_BYTES = 10 * 1024 * 1024

CLASS_NAMES = [
    "stage0_unripe",
    "stage1_ripe",
    "stage2_overripe",
    "stage3_rotten",
]

DISPLAY_NAMES = {
    "stage0_unripe": "Unripe",
    "stage1_ripe": "Ripe",
    "stage2_overripe": "Overripe",
    "stage3_rotten": "Rotten",
}

ALLOWED_FORMATS = {
    "JPEG": {".jpg", ".jpeg"},
    "PNG": {".png"},
    "WEBP": {".webp"},
    "BMP": {".bmp"},
    "GIF": {".gif"},
}


st.set_page_config(
    page_title="Banana Ripeness & Spoilage Prediction",
    page_icon="🍌",
    layout="centered",
)

st.title("🍌 Banana Ripeness & Spoilage Prediction")
st.write(
    "Upload a banana image to predict its ripeness stage using the "
    "trained MobileNetV2 model."
)


@st.cache_resource(show_spinner="Loading trained model...")
def load_trained_model():
    if not MODEL_PATH.is_file():
        raise FileNotFoundError(
            f"Trained model file not found: {MODEL_PATH.name}"
        )

    loaded_model = tf.keras.models.load_model(
        str(MODEL_PATH),
        compile=False,
    )

    if tuple(loaded_model.input_shape) != (None, IMG_SIZE, IMG_SIZE, 3):
        raise ValueError(
            f"Unexpected model input shape: {loaded_model.input_shape}"
        )

    output_shape = loaded_model.output_shape
    if (
        not isinstance(output_shape, tuple)
        or len(output_shape) != 2
        or output_shape[-1] != len(CLASS_NAMES)
    ):
        raise ValueError(
            f"Unexpected model output shape: {output_shape}"
        )

    return loaded_model


def validate_image(data, filename):
    if not data:
        raise ValueError("The uploaded image is empty.")

    if len(data) > MAX_UPLOAD_BYTES:
        raise ValueError("Image size must be 10 MB or less.")

    extension = Path(filename).suffix.lower()

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", Image.DecompressionBombWarning)

            with Image.open(io.BytesIO(data)) as image:
                image_format = image.format
                image.verify()

    except (
        UnidentifiedImageError,
        OSError,
        ValueError,
        SyntaxError,
        Image.DecompressionBombError,
        Image.DecompressionBombWarning,
    ) as exc:
        raise ValueError(
            "Invalid or unsafe image. Please upload a valid image."
        ) from exc

    if (
        image_format not in ALLOWED_FORMATS
        or extension not in ALLOWED_FORMATS[image_format]
    ):
        raise ValueError(
            "File extension does not match the image format. "
            "Upload JPEG, PNG, WEBP, BMP, or GIF."
        )


def preprocess_image(data):
    image = load_img(
        io.BytesIO(data),
        color_mode="rgb",
        target_size=(IMG_SIZE, IMG_SIZE),
        interpolation="bilinear",
    )
    array = img_to_array(image)
    array = preprocess_input(array)
    return np.expand_dims(array, axis=0)


def estimate_days(stage, temperature, humidity):
    # This is a rule-based estimate, not a validated shelf-life model.
    base_ranges = {
        "stage0_unripe": (4.0, 7.0),
        "stage1_ripe": (1.0, 3.0),
        "stage2_overripe": (0.0, 2.0),
        "stage3_rotten": (0.0, 0.0),
    }

    low, high = base_ranges[stage]
    days = (low + high) / 2.0

    if temperature > 30:
        factor = 0.5
    elif temperature > 28:
        factor = 0.65
    elif temperature > 26:
        factor = 0.8
    elif temperature < 18:
        factor = 1.3
    else:
        factor = 1.0

    if humidity > 80:
        factor *= 0.85
    elif humidity < 40:
        factor *= 0.9

    return round(max(0.0, days * factor), 1)


uploaded_file = st.file_uploader(
    "Choose a banana image",
    type=["jpg", "jpeg", "png", "webp", "bmp", "gif"],
)

col1, col2 = st.columns(2)

with col1:
    temperature = st.number_input(
        "Temperature (°C)",
        min_value=0.0,
        max_value=60.0,
        value=25.0,
        step=1.0,
    )

with col2:
    humidity = st.number_input(
        "Humidity (%)",
        min_value=0.0,
        max_value=100.0,
        value=60.0,
        step=1.0,
    )

if uploaded_file is not None:
    image_data = uploaded_file.getvalue()

    try:
        validate_image(image_data, uploaded_file.name)

        with Image.open(io.BytesIO(image_data)) as preview:
            st.image(
                preview.convert("RGB"),
                caption="Uploaded banana image",
                use_container_width=True,
            )

    except ValueError as exc:
        st.error(str(exc))
        image_data = None

if st.button("Run Prediction", type="primary", use_container_width=True):
    if uploaded_file is None:
        st.warning("Please upload an image first.")
    elif not image_data:
        st.warning("Please upload a valid image.")
    elif not (
        math.isfinite(temperature)
        and math.isfinite(humidity)
    ):
        st.error("Temperature and humidity must be valid numbers.")
    else:
        try:
            with st.spinner("Analysing image..."):
                validate_image(image_data, uploaded_file.name)
                model = load_trained_model()
                batch = preprocess_image(image_data)
                scores = np.asarray(model(batch, training=False))[0]

                if (
                    scores.shape != (len(CLASS_NAMES),)
                    or not np.isfinite(scores).all()
                ):
                    raise ValueError(
                        "The model returned invalid prediction scores."
                    )

                index = int(np.argmax(scores))
                stage = CLASS_NAMES[index]
                days_left = estimate_days(
                    stage, float(temperature), float(humidity)
                )

            st.subheader("Prediction Result")
            st.success(f"Predicted stage: **{DISPLAY_NAMES[stage]}**")
            st.metric(
                "Model output score",
                f"{float(scores[index]) * 100:.2f}%",
            )
            st.metric("Estimated days remaining", f"{days_left} days")

            st.caption(
                "The model output score is not necessarily a calibrated "
                "probability. The remaining-days figure is a rule-based "
                "estimate and has not been validated as a food-safety "
                "guarantee. Do not use this demo alone to decide whether "
                "food is safe to eat."
            )

        except Exception as exc:
            st.error(
                "Prediction failed. The model could not complete inference. "
                "Check the deployment logs for details."
            )
            print(f"Prediction error: {type(exc).__name__}: {exc}")

st.divider()
st.caption("Educational demo · MobileNetV2 · Four ripeness classes")
