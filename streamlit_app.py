
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


# --------------------------------------------------
# PAGE CONFIGURATION
# --------------------------------------------------
st.set_page_config(
    page_title="Banana AI | Ripeness & Spoilage",
    page_icon="🍌",
    layout="wide",
    initial_sidebar_state="collapsed",
)


# --------------------------------------------------
# PATHS AND MODEL SETTINGS
# --------------------------------------------------
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

STAGE_DETAILS = {
    "stage0_unripe": {
        "emoji": "🟢",
        "description": "The banana is classified as unripe.",
    },
    "stage1_ripe": {
        "emoji": "🟡",
        "description": "The banana is classified as ripe.",
    },
    "stage2_overripe": {
        "emoji": "🟤",
        "description": "The banana is classified as overripe.",
    },
    "stage3_rotten": {
        "emoji": "⚫",
        "description": "The banana is classified as rotten.",
    },
}

ALLOWED_FORMATS = {
    "JPEG": {".jpg", ".jpeg"},
    "PNG": {".png"},
    "WEBP": {".webp"},
    "BMP": {".bmp"},
    "GIF": {".gif"},
}


# --------------------------------------------------
# CUSTOM UI THEME
# --------------------------------------------------
st.markdown(
    """
    <style>
    @import url(
        'https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700;800&display=swap'
    );

    .stApp {
        background: #f6f9f4;
        color: #24372b;
        font-family: 'DM Sans', sans-serif;
    }

    .block-container {
        max-width: 1120px;
        padding-top: 1.5rem;
        padding-bottom: 3rem;
    }

    [data-testid="stHeader"] {
        background: rgba(246, 249, 244, 0.9);
    }

    .hero {
        background: linear-gradient(125deg, #174b2b, #2d7945);
        padding: 34px 36px;
        border-radius: 22px;
        color: white;
        margin-bottom: 26px;
        box-shadow: 0 10px 28px rgba(31, 83, 48, 0.12);
    }

    .hero-tag {
        display: inline-block;
        padding: 6px 12px;
        border: 1px solid rgba(255,255,255,0.35);
        border-radius: 30px;
        font-size: 12px;
        letter-spacing: 1px;
        text-transform: uppercase;
        margin-bottom: 14px;
        color: #e4f6e7;
    }

    .hero h1 {
        color: white !important;
        font-size: clamp(27px, 4vw, 42px) !important;
        line-height: 1.2;
        margin: 0 0 12px 0;
        font-weight: 800 !important;
        letter-spacing: -1px;
    }

    .hero p {
        color: #e1eee3;
        font-size: 15px;
        line-height: 1.7;
        margin: 0;
        max-width: 720px;
    }

    .section-title {
        font-size: 22px;
        font-weight: 800;
        color: #214f31;
        margin: 8px 0 4px 0;
    }

    .section-description {
        color: #68796b;
        font-size: 14px;
        margin-bottom: 18px;
    }

    .info-strip {
        background: #eaf3e8;
        border: 1px solid #d5e6d3;
        padding: 15px 18px;
        border-radius: 14px;
        color: #365b3e;
        font-size: 13px;
        line-height: 1.7;
        margin: 16px 0 22px 0;
    }

    [data-testid="stFileUploader"] {
        background: white;
        border: 1px dashed #8bb79a;
        border-radius: 16px;
        padding: 15px;
    }

    [data-testid="stFileUploader"] section {
        background: transparent;
    }

    [data-testid="stNumberInput"] {
        background: white;
        border-radius: 12px;
    }

    [data-testid="stNumberInput"] input {
        color: #23432d;
        font-weight: 600;
    }

    .stButton > button {
        min-height: 48px;
        border-radius: 12px;
        background: #267344;
        border: 1px solid #267344;
        color: white;
        font-weight: 700;
        transition: all 0.2s ease;
    }

    .stButton > button:hover {
        background: #195a32;
        border-color: #195a32;
        color: white;
    }

    [data-testid="stMetric"] {
        background: white;
        border: 1px solid #dfe9dd;
        border-radius: 15px;
        padding: 18px;
    }

    [data-testid="stMetricLabel"] {
        color: #627365;
    }

    [data-testid="stMetricValue"] {
        color: #215c36;
        font-weight: 800;
    }

    [data-testid="stImage"] img {
        border-radius: 16px;
    }

    .result-card {
        background: white;
        border: 1px solid #dfe9dd;
        border-radius: 18px;
        padding: 24px;
        margin: 12px 0;
    }

    .result-label {
        color: #708071;
        font-size: 12px;
        text-transform: uppercase;
        letter-spacing: 1.2px;
        font-weight: 700;
    }

    .result-stage {
        color: #215c36;
        font-size: 30px;
        font-weight: 800;
        margin: 8px 0;
    }

    .result-description {
        color: #607263;
        font-size: 14px;
        line-height: 1.7;
    }

    .recommendation-card {
        background: white;
        border: 1px solid #dfe9dd;
        border-radius: 16px;
        padding: 18px 22px;
        margin: 10px 0;
    }

    .recommendation-heading {
        color: #215c36;
        font-size: 16px;
        font-weight: 800;
        margin-bottom: 8px;
    }

    .footer {
        text-align: center;
        color: #788779;
        font-size: 12px;
        padding-top: 18px;
        line-height: 1.8;
    }

    hr {
        border-color: #dce7da;
        margin: 28px 0;
    }

    @media (max-width: 640px) {
        .block-container {
            padding-top: 1rem;
            padding-left: 1rem;
            padding-right: 1rem;
        }

        .hero {
            padding: 25px 22px;
            border-radius: 17px;
        }

        .result-stage {
            font-size: 25px;
        }
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# --------------------------------------------------
# HEADER
# --------------------------------------------------
st.markdown(
    """
    <div class="hero">
        <div class="hero-tag">AI-powered educational demo</div>
        <h1>Banana Ripeness<br> &amp; Spoilage Prediction</h1>
        <p>
            Explore banana ripeness classification using a trained
            MobileNetV2 image model. Upload an image, provide storage
            conditions, and view the model's predicted stage along
            with practical storage recommendations.
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="info-strip">
        <strong>How it works</strong><br>
        01. Upload a banana image &nbsp; → &nbsp;
        02. Enter temperature and humidity &nbsp; → &nbsp;
        03. Run the prediction and view recommendations
    </div>
    """,
    unsafe_allow_html=True,
)


# --------------------------------------------------
# MODEL LOADING
# --------------------------------------------------
@st.cache_resource(show_spinner="Loading trained MobileNetV2 model...")
def load_trained_model():
    if not MODEL_PATH.is_file():
        raise FileNotFoundError(
            f"Trained model file not found: {MODEL_PATH.name}"
        )

    loaded_model = tf.keras.models.load_model(
        str(MODEL_PATH),
        compile=False,
    )

    if tuple(loaded_model.input_shape) != (
        None,
        IMG_SIZE,
        IMG_SIZE,
        3,
    ):
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


# --------------------------------------------------
# IMAGE VALIDATION
# --------------------------------------------------
def validate_image(data, filename):
    if not data:
        raise ValueError("The uploaded image is empty.")

    if len(data) > MAX_UPLOAD_BYTES:
        raise ValueError("Image size must be 10 MB or less.")

    extension = Path(filename).suffix.lower()

    try:
        with warnings.catch_warnings():
            warnings.simplefilter(
                "error",
                Image.DecompressionBombWarning,
            )

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


# --------------------------------------------------
# IMAGE PREPROCESSING
# Keep preprocessing consistent with the original app.
# --------------------------------------------------
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


# --------------------------------------------------
# RULE-BASED SHELF-LIFE ESTIMATE
# Not a validated shelf-life prediction model.
# --------------------------------------------------
def estimate_days(stage, temperature, humidity):
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


# --------------------------------------------------
# SMART STORAGE RECOMMENDATIONS
# General guidance based on the predicted stage
# and user-entered storage conditions.
# --------------------------------------------------
def get_recommendations(stage, temperature, humidity):
    recommendations = {
        "stage0_unripe": [
            (
                "Ripening: Allow the banana to ripen at room "
                "temperature if you want it to ripen sooner."
            ),
            (
                "Storage: Keep it in a clean, ventilated place "
                "away from direct sunlight and excessive heat."
            ),
            (
                "Monitoring: Check the banana regularly for "
                "changes in colour, texture, and smell."
            ),
        ],
        "stage1_ripe": [
            (
                "Consumption: Prioritize consuming or processing "
                "the banana while it is ripe."
            ),
            (
                "Storage: Keep it away from direct sunlight and "
                "excessive heat."
            ),
            (
                "Monitoring: Inspect regularly for soft spots, "
                "leakage, mold, or unusual odour."
            ),
        ],
        "stage2_overripe": [
            (
                "Inspection: Carefully inspect for mold, leakage, "
                "or unusual odour before handling."
            ),
            (
                "Use: If the banana appears sound and shows no "
                "signs of spoilage, it may be suitable for cooking "
                "or baking."
            ),
            (
                "Caution: Discard it if mold, unusual odour, "
                "leakage, or other signs of spoilage are present."
            ),
        ],
        "stage3_rotten": [
            (
                "Safety: Do not consume a banana classified as "
                "rotten or showing signs of spoilage."
            ),
            (
                "Disposal: Discard the fruit and clean any surfaces "
                "contaminated by leaking material."
            ),
            (
                "Important: Do not rely on the AI prediction alone "
                "to determine whether food is safe to eat."
            ),
        ],
    }

    # Recommendations for the predicted stage
    tips = list(recommendations[stage])

    # Temperature-related suggestions
    if stage != "stage3_rotten":
        if temperature > 30:
            tips.append(
                "Temperature: The entered temperature is high. "
                "Keep the fruit away from heat and direct sunlight."
            )
        elif temperature < 18:
            tips.append(
                "Temperature: Ripening may be slower in cooler "
                "conditions. Continue inspecting the fruit regularly."
            )

        # Humidity-related suggestions
        if humidity > 80:
            tips.append(
                "Humidity: High humidity can encourage surface "
                "moisture. Maintain ventilation and avoid condensation."
            )
        elif humidity < 40:
            tips.append(
                "Humidity: The entered humidity is low. Check the "
                "fruit for shrivelling or signs of moisture loss."
            )

    return tips


# --------------------------------------------------
# MAIN INPUT AREA
# --------------------------------------------------
st.markdown(
    '<div class="section-title">01 · Upload your image</div>',
    unsafe_allow_html=True,
)

st.markdown(
    '<div class="section-description">'
    'Choose a clear banana photo for image classification.'
    '</div>',
    unsafe_allow_html=True,
)

uploaded_file = st.file_uploader(
    "Choose a banana image",
    type=["jpg", "jpeg", "png", "webp", "bmp", "gif"],
    help="Supported formats: JPG, JPEG, PNG, WEBP, BMP, GIF. Maximum 10 MB.",
)

image_data = None
valid_image = False

if uploaded_file is not None:
    image_data = uploaded_file.getvalue()

    try:
        validate_image(image_data, uploaded_file.name)

        with Image.open(io.BytesIO(image_data)) as preview:
            preview_image = preview.convert("RGB")

        image_col, details_col = st.columns(
            [1.15, 0.85],
            gap="large",
        )

        with image_col:
            st.image(
                preview_image,
                caption="Selected banana image",
                use_container_width=True,
            )

        with details_col:
            st.markdown("#### Image ready")
            st.write("Your image passed the basic file validation checks.")
            st.caption(f"File: {uploaded_file.name}")
            st.caption(
                f"Size: {len(image_data) / (1024 * 1024):.2f} MB"
            )
            st.caption(
                "File validation checks format and size, "
                "not whether the image contains a banana."
            )

        valid_image = True

    except ValueError as exc:
        st.error(str(exc))
        image_data = None


st.divider()


# --------------------------------------------------
# STORAGE CONDITIONS
# --------------------------------------------------
st.markdown(
    '<div class="section-title">02 · Storage conditions</div>',
    unsafe_allow_html=True,
)

st.markdown(
    '<div class="section-description">'
    'Enter the temperature and humidity values to see the demo estimate.'
    '</div>',
    unsafe_allow_html=True,
)

temp_col, humidity_col = st.columns(2, gap="large")

with temp_col:
    temperature = st.number_input(
        "Temperature (°C)",
        min_value=0.0,
        max_value=60.0,
        value=25.0,
        step=1.0,
        help="Enter the storage temperature in degrees Celsius.",
    )

with humidity_col:
    humidity = st.number_input(
        "Relative humidity (%)",
        min_value=0.0,
        max_value=100.0,
        value=60.0,
        step=1.0,
        help="Enter relative humidity as a percentage.",
    )


st.divider()


# --------------------------------------------------
# PREDICTION
# --------------------------------------------------
st.markdown(
    '<div class="section-title">03 · Prediction</div>',
    unsafe_allow_html=True,
)

st.markdown(
    '<div class="section-description">'
    'Run the trained image model to classify the ripeness stage.'
    '</div>',
    unsafe_allow_html=True,
)

run_prediction = st.button(
    "🍌  Run Prediction",
    type="primary",
    use_container_width=True,
)

if run_prediction:
    if uploaded_file is None:
        st.warning("Please upload a banana image first.")

    elif not valid_image or not image_data:
        st.warning("Please upload a valid image before predicting.")

    elif not (
        math.isfinite(temperature)
        and math.isfinite(humidity)
    ):
        st.error("Temperature and humidity must be valid numbers.")

    else:
        try:
            with st.spinner("Analysing image with MobileNetV2..."):
                validate_image(image_data, uploaded_file.name)

                model = load_trained_model()
                batch = preprocess_image(image_data)

                scores = np.asarray(
                    model(batch, training=False)
                )[0]

                if (
                    scores.shape != (len(CLASS_NAMES),)
                    or not np.isfinite(scores).all()
                ):
                    raise ValueError(
                        "The model returned invalid prediction scores."
                    )

                index = int(np.argmax(scores))
                stage = CLASS_NAMES[index]

                # Existing rule-based estimate
                days_left = estimate_days(
                    stage,
                    float(temperature),
                    float(humidity),
                )

                # New smart recommendations
                recommendations = get_recommendations(
                    stage,
                    float(temperature),
                    float(humidity),
                )

            # ------------------------------------------
            # PREDICTION RESULTS
            # ------------------------------------------
            st.markdown(
                '<div class="section-title">Prediction results</div>',
                unsafe_allow_html=True,
            )

            result = STAGE_DETAILS[stage]

            st.markdown(
                f"""
                <div class="result-card">
                    <div class="result-label">Predicted ripeness stage</div>
                    <div class="result-stage">
                        {result["emoji"]} {DISPLAY_NAMES[stage]}
                    </div>
                    <div class="result-description">
                        {result["description"]}
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

            score_col, days_col = st.columns(2, gap="large")

            with score_col:
                st.metric(
                    "Model output score",
                    f"{float(scores[index]) * 100:.2f}%",
                )

            with days_col:
                st.metric(
                    "Estimated days remaining",
                    f"{days_left} days",
                )

            # ------------------------------------------
            # NEW: SMART RECOMMENDATIONS
            # ------------------------------------------
            st.divider()

            st.markdown(
                '<div class="section-title">'
                '04 · Smart Storage Recommendations</div>',
                unsafe_allow_html=True,
            )

            st.markdown(
                '<div class="section-description">'
                'Suggestions based on the predicted stage and '
                'the temperature and humidity you entered.'
                '</div>',
                unsafe_allow_html=True,
            )

            st.markdown(
                '<div class="recommendation-card">'
                '<div class="recommendation-heading">'
                'Recommended next steps</div>'
                '</div>',
                unsafe_allow_html=True,
            )

            for tip in recommendations:
                st.markdown(f"- {tip}")

            st.info(
                "These are general handling suggestions, not "
                "personalized or experimentally validated storage advice."
            )

            # ------------------------------------------
            # INTERPRETATION AND SAFETY NOTE
            # ------------------------------------------
            st.caption(
                "Interpretation note: The model output score is not "
                "necessarily a calibrated probability or a verified "
                "accuracy measurement. The remaining-days value is a "
                "rule-based estimate, not an independently validated "
                "shelf-life model. This demo cannot determine food safety. "
                "Do not use its result alone to decide whether food is "
                "safe to eat."
            )

        except Exception as exc:
            st.error(
                "Prediction failed. The model could not complete inference. "
                "Check the deployment logs for details."
            )

            print(
                f"Prediction error: {type(exc).__name__}: {exc}"
            )


# --------------------------------------------------
# FOOTER
# --------------------------------------------------
st.divider()

st.markdown(
    """
    <div class="footer">
        <strong>Banana Ripeness &amp; Spoilage Prediction</strong><br>
        Educational demonstration · MobileNetV2 · Four ripeness classes
        <br><br>
        Predictions are model outputs, not guarantees of food quality
        or safety.
    </div>
    """,
    unsafe_allow_html=True,
)
