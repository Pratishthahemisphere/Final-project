"""Streamlit Web Application for Venomous vs Non-Venomous Snake Detection with Grad-CAM.

Run with:
    streamlit run app/streamlit_app.py
"""

import sys
from pathlib import Path

# Ensure repository root is in Python module search path
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import streamlit as st
from PIL import Image

from src.explain import GradCAM, extract_cbam_spatial_attention, overlay_heatmap
from src.predict import SnakeClassifier

# -----------------------------------------------------------------------------
# Page Configuration
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="Snake Venomous Detection | Attention Network",
    page_icon="🐍",
    layout="wide",
    initial_sidebar_state="expanded",
)

# -----------------------------------------------------------------------------
# Prominent Safety Notice (Mandatory)
# -----------------------------------------------------------------------------
st.markdown(
    """
    <div style="background-color: #fff3cd; border-left: 6px solid #ffc107; padding: 15px; border-radius: 4px; margin-bottom: 25px;">
        <h4 style="color: #856404; margin-top: 0; margin-bottom: 8px;">⚠️ CRITICAL SAFETY NOTICE</h4>
        <p style="color: #856404; font-size: 15px; margin-bottom: 0; line-height: 1.5;">
            <strong>This is an educational ML demo and can be wrong.</strong> Never use it to decide whether
            to touch, handle or approach a snake. Treat every unidentified snake as dangerous and contact
            local wildlife authorities or emergency medical services immediately.
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)

st.title("🐍 Venomous vs Non-Venomous Snake Detection")
st.markdown(
    "Deep Residual Attention Network (ResNet50 + CBAM) with clinical uncertainty gating and Grad-CAM interpretability."
)

# -----------------------------------------------------------------------------
# Sidebar: Configuration & Model Loading
# -----------------------------------------------------------------------------
st.sidebar.header("⚙️ Model Configuration")

checkpoint_input = st.sidebar.text_input(
    "Checkpoint Path",
    value="outputs/checkpoints/best.pt",
    help="Path to the saved PyTorch weights file (.pt)",
)

checkpoint_file = Path(checkpoint_input)
if not checkpoint_file.is_file():
    st.sidebar.error("❌ Checkpoint file not found!")
    st.error(
        f"**Checkpoint not found at:** `{checkpoint_file.resolve()}`\n\n"
        "Please train a model first before running inference:\n\n"
        "```bash\npython -m src.train --config configs/config.yaml\n```"
    )
    st.stop()


@st.cache_resource(show_spinner="Loading Attention Network into memory...")
def load_cached_classifier(ckpt_path: str) -> SnakeClassifier:
    """Load and cache the classifier resource to ensure single model instantiation."""
    return SnakeClassifier(checkpoint_path=ckpt_path)


try:
    classifier = load_cached_classifier(str(checkpoint_file))
    st.sidebar.success(f"Loaded: `{checkpoint_file.name}`")
except Exception as e:  # noqa: BLE001
    st.sidebar.error(f"Failed to load checkpoint: {e}")
    st.error(f"Error initializing SnakeClassifier: {e}")
    st.stop()

# Sidebar Hyperparameters
threshold = st.sidebar.slider(
    "Uncertainty Threshold",
    min_value=0.50,
    max_value=0.95,
    value=float(classifier.uncertainty_threshold),
    step=0.01,
    help="Predictions with confidence below this value will be flagged as 'uncertain'.",
)

gradcam_alpha = st.sidebar.slider(
    "Grad-CAM Heatmap Opacity",
    min_value=0.10,
    max_value=0.80,
    value=0.45,
    step=0.05,
    help="Blend ratio between the original image and the visual attention heatmap.",
)

show_cbam_attention = st.sidebar.checkbox(
    "Show CBAM Spatial Attention Map",
    value=True,
    help="Display the raw spatial attention mask extracted from CBAM layer 4.",
)

# -----------------------------------------------------------------------------
# Main Application: File Upload & Processing
# -----------------------------------------------------------------------------
uploaded_files = st.file_uploader(
    "Upload Snake Image(s)",
    type=["jpg", "jpeg", "png", "webp"],
    accept_multiple_files=True,
    help="Upload one or more images in JPG, PNG, or WebP format.",
)

if uploaded_files:
    st.subheader(f"Results ({len(uploaded_files)} image{'s' if len(uploaded_files) > 1 else ''})")

    for file_idx, uploaded_file in enumerate(uploaded_files):
        st.markdown("---")
        # Graceful error handling for corrupted uploads
        try:
            image = Image.open(uploaded_file).convert("RGB")
        except Exception as err:  # noqa: BLE001
            st.error(f"❌ Failed to decode '{uploaded_file.name}': {err}. Skipping this file.")
            continue

        # Run inference
        pred = classifier.predict(image, uncertainty_threshold=threshold)

        label = pred["label"]
        confidence = pred["confidence"]
        p_venomous = pred["probabilities"]["venomous"]
        p_non_venomous = pred["probabilities"]["non_venomous"]

        # Color-coded badge representation
        if label == "venomous":
            badge_html = f"""
            <div style="background-color: #f8d7da; border: 2px solid #dc3545; color: #721c24;
                        padding: 12px; border-radius: 6px; font-weight: bold; font-size: 18px; text-align: center;">
                🚨 VENOMOUS SNAKE ({confidence * 100:.1f}% Confidence)
            </div>
            """
        elif label == "non_venomous":
            badge_html = f"""
            <div style="background-color: #d4edda; border: 2px solid #28a745; color: #155724;
                        padding: 12px; border-radius: 6px; font-weight: bold; font-size: 18px; text-align: center;">
                ✅ NON-VENOMOUS SNAKE ({confidence * 100:.1f}% Confidence)
            </div>
            """
        else:  # uncertain
            badge_html = f"""
            <div style="background-color: #fff3cd; border: 2px solid #ffc107; color: #856404;
                        padding: 12px; border-radius: 6px; font-weight: bold; font-size: 18px; text-align: center;">
                ⚠️ UNCERTAIN PREDICTION ({confidence * 100:.1f}% < {threshold * 100:.0f}% Threshold)
            </div>
            """

        cols = st.columns([1.2, 1.3, 1.5])

        with cols[0]:
            st.markdown(f"**Image:** `{uploaded_file.name}`")
            st.image(image, use_container_width=True)

        with cols[1]:
            st.markdown("### Classification Output")
            st.markdown(badge_html, unsafe_allow_html=True)
            st.write("")

            st.write("**Probability Distribution:**")
            st.write(f"• **Venomous:** `{p_venomous * 100:.2f}%`")
            st.progress(min(1.0, max(0.0, p_venomous)))

            st.write(f"• **Non-Venomous:** `{p_non_venomous * 100:.2f}%`")
            st.progress(min(1.0, max(0.0, p_non_venomous)))

            if pred["uncertain"]:
                st.info(
                    f"ℹ️ Confidence ({confidence * 100:.1f}%) is below the {threshold * 100:.0f}% safety cutoff. "
                    "In clinical and field scenarios, uncertain classifications must be referred to a human specialist."
                )

        with cols[2]:
            st.markdown("### Visual Attention Explanations")
            # Generate Grad-CAM heatmap
            input_tensor = classifier.transform(image).unsqueeze(0).to(classifier.device)
            with GradCAM(classifier.model, classifier.model.get_target_layer()) as cam_engine:
                gradcam_map, _, _ = cam_engine.generate_heatmap(input_tensor)

            gradcam_overlay = overlay_heatmap(image, gradcam_map, alpha=gradcam_alpha)
            st.image(
                gradcam_overlay,
                caption="Grad-CAM Activation (Features informing classification)",
                use_container_width=True,
            )

            if show_cbam_attention:
                cbam_map = extract_cbam_spatial_attention(classifier.model, input_tensor)
                cbam_overlay = overlay_heatmap(image, cbam_map, alpha=gradcam_alpha)
                st.image(
                    cbam_overlay,
                    caption="CBAM Layer 4 Spatial Attention Focus",
                    use_container_width=True,
                )
else:
    st.info("👆 Upload one or more snake images above to analyze venomous status and view attention heatmaps.")
