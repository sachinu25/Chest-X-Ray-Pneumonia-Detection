"""
Chest X-Ray Pneumonia Detection — Production Streamlit Dashboard.

A professional, healthcare-themed web application for real-time pneumonia
classification from chest X-ray images with Grad-CAM explainability,
interactive charts, prediction history, and PDF report generation.
"""

import os
import sys
from datetime import datetime
from pathlib import Path

import streamlit as st
import torch
import numpy as np
from PIL import Image

# ---------------------------------------------------------------------------
# Ensure project root is on sys.path so `xray.*` and `utils.*` imports work
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from utils.model_utils import (
    load_model,
    preprocess_image,
    predict,
    get_risk_level,
    DEVICE,
    DEFAULT_MODEL_PATH,
)
from utils.gradcam import GradCAMGenerator, create_gradcam_overlay
from utils.pdf_report import generate_pdf_report
from utils.logger import get_logger

logger = get_logger("streamlit_app")

# ---------------------------------------------------------------------------
# Page Config (MUST be the first Streamlit command)
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Chest X-Ray AI Dashboard",
    page_icon="🫁",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Load Custom CSS
# ---------------------------------------------------------------------------
CSS_PATH = PROJECT_ROOT / "assets" / "style.css"
if CSS_PATH.exists():
    with open(CSS_PATH) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Cached Model Loading
# ---------------------------------------------------------------------------
@st.cache_resource(show_spinner=False)
def get_model():
    """Load and cache the trained model."""
    try:
        model = load_model()
        logger.info(f"Model loaded successfully on {DEVICE}")
        return model
    except FileNotFoundError as e:
        logger.error(f"Model file not found: {e}")
        return None
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        return None


# ---------------------------------------------------------------------------
# Session State Initialization
# ---------------------------------------------------------------------------
if "history" not in st.session_state:
    st.session_state.history = []


# ---------------------------------------------------------------------------
# Helper: Render a metric card via HTML
# ---------------------------------------------------------------------------
def metric_card(icon: str, label: str, value: str, accent_class: str = ""):
    """Render a glassmorphic metric card."""
    st.markdown(
        f"""
        <div class="metric-card {accent_class}">
            <div class="metric-icon">{icon}</div>
            <div class="metric-label">{label}</div>
            <div class="metric-value">{value}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def section_header(icon: str, title: str):
    """Render a styled section header."""
    st.markdown(
        f'<div class="section-header">{icon} {title}</div>',
        unsafe_allow_html=True,
    )


def custom_divider():
    st.markdown('<div class="custom-divider"></div>', unsafe_allow_html=True)


# ===================================================================
#  SIDEBAR
# ===================================================================
with st.sidebar:
    # Logo / Title
    st.markdown(
        """
        <div style="text-align: center; padding: 1.2rem 0 0.5rem 0;">
            <div style="font-size: 3rem; margin-bottom: 0.2rem;">🫁</div>
            <div style="font-size: 1.15rem; font-weight: 700; color: #0EA5B5;
                        letter-spacing: 0.5px;">Chest X-Ray AI</div>
            <div style="font-size: 0.72rem; color: #9AA0A6; margin-top: 0.2rem;
                        text-transform: uppercase; letter-spacing: 1.5px;">
                Pneumonia Detection System
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    custom_divider()

    page = st.radio(
        "NAVIGATION",
        ["🏠 Overview", "🔬 Predict", "📜 History", "ℹ️ About"],
        label_visibility="collapsed",
    )

    custom_divider()

    # Device info
    device_icon = "🟢" if torch.cuda.is_available() else "🟡"
    device_name = (
        torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
    )
    st.markdown(
        f"""
        <div style="background: rgba(255,255,255,0.03); border: 1px solid rgba(255,255,255,0.06);
                    border-radius: 8px; padding: 0.8rem; margin-top: 0.5rem;">
            <div style="font-size: 0.7rem; color: #9AA0A6; text-transform: uppercase;
                        letter-spacing: 1px; margin-bottom: 0.4rem;">Compute Device</div>
            <div style="font-size: 0.85rem; color: #E8EAED;">
                {device_icon} {device_name}
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Model status
    model = get_model()
    model_status = "✅ Loaded" if model is not None else "❌ Not Found"
    model_color = "#00E676" if model is not None else "#FF1744"
    st.markdown(
        f"""
        <div style="background: rgba(255,255,255,0.03); border: 1px solid rgba(255,255,255,0.06);
                    border-radius: 8px; padding: 0.8rem; margin-top: 0.5rem;">
            <div style="font-size: 0.7rem; color: #9AA0A6; text-transform: uppercase;
                        letter-spacing: 1px; margin-bottom: 0.4rem;">Model Status</div>
            <div style="font-size: 0.85rem; color: {model_color};">
                {model_status}
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <div style="position: fixed; bottom: 1rem; padding: 0.5rem;
                    font-size: 0.65rem; color: #5F6368; text-align: center;">
            v2.0 &bull; Built with Streamlit &amp; PyTorch
        </div>
        """,
        unsafe_allow_html=True,
    )

# ===================================================================
#  PAGES
# ===================================================================

# -------------------------------------------------------------------
#  OVERVIEW PAGE
# -------------------------------------------------------------------
if page == "🏠 Overview":
    st.markdown(
        """
        <div style="text-align: center; padding: 1rem 0 0.5rem 0;" class="animate-in">
            <h1 style="font-size: 2.2rem; font-weight: 800; margin-bottom: 0.3rem;
                       background: linear-gradient(135deg, #0A7E8C, #0EA5B5);
                       -webkit-background-clip: text; -webkit-text-fill-color: transparent;
                       background-clip: text;">
                Chest X-Ray Pneumonia Detection
            </h1>
            <p style="color: #9AA0A6; font-size: 1rem; max-width: 600px; margin: 0 auto;">
                AI-powered analysis of chest radiographs for rapid pneumonia screening
                using deep learning with ResNet-18 transfer learning.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    custom_divider()
    section_header("📊", "Model Performance Metrics")

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        metric_card("🎯", "Test Accuracy", "92.79%", "accuracy")
    with col2:
        metric_card("📏", "Macro F1 Score", "0.9197", "f1")
    with col3:
        metric_card("📈", "ROC-AUC", "0.9923", "roc")
    with col4:
        metric_card("🧠", "Architecture", "ResNet-18", "dataset")

    custom_divider()
    section_header("📋", "Dataset Summary")

    col1, col2, col3 = st.columns(3)
    with col1:
        metric_card("📁", "Total Images", "5,232", "dataset")
    with col2:
        metric_card("🟢", "Normal Images", "1,349", "accuracy")
    with col3:
        metric_card("🔴", "Pneumonia Images", "3,883", "f1")

    custom_divider()
    section_header("⚙️", "Training Configuration")

    col1, col2 = st.columns(2)
    with col1:
        st.markdown(
            """
            <div class="info-box">
                <strong>Training Details</strong><br>
                • <strong>Backbone:</strong> ResNet-18 (ImageNet pretrained)<br>
                • <strong>Epochs:</strong> 25 (with early stopping)<br>
                • <strong>Optimizer:</strong> Adam (lr=0.0003, wd=1e-4)<br>
                • <strong>Scheduler:</strong> StepLR (step=7, γ=0.5)<br>
                • <strong>Loss:</strong> Weighted CrossEntropy
            </div>
            """,
            unsafe_allow_html=True,
        )
    with col2:
        st.markdown(
            """
            <div class="info-box">
                <strong>Data Augmentation</strong><br>
                • Random Horizontal Flip<br>
                • Random Rotation (±15°)<br>
                • Color Jitter (brightness, contrast, saturation, hue)<br>
                • Random Affine (translate, scale)<br>
                • ImageNet Normalization
            </div>
            """,
            unsafe_allow_html=True,
        )

    custom_divider()

    # Per-class metrics table
    section_header("📊", "Per-Class Performance")

    st.markdown(
        """
        <div style="overflow-x: auto;">
        <table style="width: 100%; border-collapse: collapse; margin: 0.5rem 0;
                      background: rgba(26,29,41,0.85); border-radius: 8px; overflow: hidden;">
            <thead>
                <tr style="background: rgba(10,126,140,0.15);">
                    <th style="padding: 0.75rem 1rem; text-align: left; color: #0EA5B5;
                               font-size: 0.78rem; text-transform: uppercase; letter-spacing: 1px;">
                        Class</th>
                    <th style="padding: 0.75rem 1rem; text-align: center; color: #0EA5B5;
                               font-size: 0.78rem; text-transform: uppercase; letter-spacing: 1px;">
                        Precision</th>
                    <th style="padding: 0.75rem 1rem; text-align: center; color: #0EA5B5;
                               font-size: 0.78rem; text-transform: uppercase; letter-spacing: 1px;">
                        Recall</th>
                    <th style="padding: 0.75rem 1rem; text-align: center; color: #0EA5B5;
                               font-size: 0.78rem; text-transform: uppercase; letter-spacing: 1px;">
                        F1 Score</th>
                    <th style="padding: 0.75rem 1rem; text-align: center; color: #0EA5B5;
                               font-size: 0.78rem; text-transform: uppercase; letter-spacing: 1px;">
                        Support</th>
                </tr>
            </thead>
            <tbody>
                <tr style="border-bottom: 1px solid rgba(255,255,255,0.06);">
                    <td style="padding: 0.65rem 1rem; color: #00E676; font-weight: 600;">
                        🟢 NORMAL</td>
                    <td style="padding: 0.65rem 1rem; text-align: center; color: #E8EAED;">
                        0.8923</td>
                    <td style="padding: 0.65rem 1rem; text-align: center; color: #E8EAED;">
                        0.8846</td>
                    <td style="padding: 0.65rem 1rem; text-align: center; color: #E8EAED;">
                        0.8885</td>
                    <td style="padding: 0.65rem 1rem; text-align: center; color: #9AA0A6;">
                        234</td>
                </tr>
                <tr>
                    <td style="padding: 0.65rem 1rem; color: #FF1744; font-weight: 600;">
                        🔴 PNEUMONIA</td>
                    <td style="padding: 0.65rem 1rem; text-align: center; color: #E8EAED;">
                        0.9464</td>
                    <td style="padding: 0.65rem 1rem; text-align: center; color: #E8EAED;">
                        0.9513</td>
                    <td style="padding: 0.65rem 1rem; text-align: center; color: #E8EAED;">
                        0.9488</td>
                    <td style="padding: 0.65rem 1rem; text-align: center; color: #9AA0A6;">
                        390</td>
                </tr>
            </tbody>
        </table>
        </div>
        """,
        unsafe_allow_html=True,
    )


# -------------------------------------------------------------------
#  PREDICT PAGE
# -------------------------------------------------------------------
elif page == "🔬 Predict":
    st.markdown(
        """
        <div class="animate-in">
            <h1 style="font-size: 1.8rem; font-weight: 700;
                       background: linear-gradient(135deg, #0A7E8C, #0EA5B5);
                       -webkit-background-clip: text; -webkit-text-fill-color: transparent;
                       background-clip: text; margin-bottom: 0.2rem;">
                X-Ray Analysis
            </h1>
            <p style="color: #9AA0A6; font-size: 0.9rem;">
                Upload a chest X-ray image to get an AI-powered pneumonia assessment
                with explainability visualization.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    custom_divider()

    if model is None:
        st.error(
            "⚠️ Model could not be loaded. Please ensure `xray_model.pth` exists "
            "in the project root directory."
        )
        st.stop()

    # File uploader
    uploaded_file = st.file_uploader(
        "Upload a Chest X-Ray Image",
        type=["jpg", "jpeg", "png"],
        help="Supported formats: JPG, JPEG, PNG. Max size: 10 MB.",
        key="xray_uploader",
    )

    if uploaded_file is not None:
        # Validate file size
        file_size = uploaded_file.size
        if file_size > 10 * 1024 * 1024:
            st.error("❌ File too large. Please upload an image under 10 MB.")
            st.stop()

        try:
            image = Image.open(uploaded_file).convert("RGB")
        except Exception:
            st.error(
                "❌ Could not open the uploaded file as an image. "
                "Please upload a valid chest X-ray."
            )
            logger.warning(f"Invalid image upload: {uploaded_file.name}")
            st.stop()

        # Layout: image preview + info
        col_img, col_info = st.columns([2, 1])
        with col_img:
            st.image(image, caption="Uploaded X-Ray", use_container_width=True)
        with col_info:
            st.markdown(
                f"""
                <div class="info-box">
                    <strong>Image Details</strong><br>
                    📄 <strong>File:</strong> {uploaded_file.name}<br>
                    📐 <strong>Size:</strong> {image.size[0]} × {image.size[1]} px<br>
                    💾 <strong>File Size:</strong> {file_size / 1024:.1f} KB<br>
                    🕐 <strong>Uploaded:</strong> {datetime.now().strftime('%H:%M:%S')}
                </div>
                """,
                unsafe_allow_html=True,
            )

        custom_divider()

        # Predict button
        if st.button("🔬 Analyze X-Ray", use_container_width=True, type="primary"):
            with st.spinner("🧠 Analyzing chest X-ray..."):
                try:
                    # Preprocess & predict
                    tensor = preprocess_image(image)
                    label, confidence, prob_dict = predict(model, tensor)
                    risk_level, risk_color = get_risk_level(label, confidence)

                    logger.info(
                        f"Prediction: {label} ({confidence:.4f}) | "
                        f"Risk: {risk_level} | File: {uploaded_file.name}"
                    )

                except Exception as e:
                    st.error(f"❌ Prediction failed: {e}")
                    logger.error(f"Prediction error: {e}", exc_info=True)
                    st.stop()

            # ---- RESULTS ----
            card_class = "pneumonia" if label == "PNEUMONIA" else "normal"
            label_emoji = "🔴" if label == "PNEUMONIA" else "🟢"

            st.markdown(
                f"""
                <div class="prediction-card {card_class} animate-in">
                    <div style="font-size: 0.85rem; color: #9AA0A6; text-transform: uppercase;
                                letter-spacing: 1.5px; margin-bottom: 0.5rem;">
                        AI Diagnosis Result
                    </div>
                    <div class="prediction-label {card_class}">
                        {label_emoji} {label}
                    </div>
                    <div class="confidence-value">{confidence * 100:.1f}%</div>
                    <div style="font-size: 0.8rem; color: #9AA0A6; margin: 0.3rem 0 0.8rem 0;">
                        Confidence Score
                    </div>
                    <div class="risk-badge" style="background: {risk_color}20; color: {risk_color};
                                border: 1px solid {risk_color};">
                        {risk_level}
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

            custom_divider()

            # ---- CHARTS ----
            section_header("📊", "Probability Distribution")

            import altair as alt
            import pandas as pd

            chart_data = pd.DataFrame({
                "Class": list(prob_dict.keys()),
                "Probability (%)": [val * 100 for val in prob_dict.values()]
            })

            chart = (
                alt.Chart(chart_data)
                .mark_bar(cornerRadiusEnd=6, height=32)
                .encode(
                    x=alt.X("Probability (%)", scale=alt.Scale(domain=[0, 100])),
                    y=alt.Y("Class", sort=None, axis=alt.Axis(labelFontSize=12, labelColor="#E8EAED", title=None)),
                    color=alt.Color(
                        "Class",
                        scale=alt.Scale(
                            domain=["NORMAL", "PNEUMONIA"],
                            range=["#00E676", "#FF1744"]
                        ),
                        legend=None
                    ),
                    tooltip=["Class", alt.Tooltip("Probability (%)", format=".1f")]
                )
                .properties(height=140)
            )

            st.altair_chart(chart, use_container_width=True)

            custom_divider()

            # ---- GRAD-CAM ----
            section_header("🔍", "Grad-CAM Explainability")

            gradcam_image = None
            try:
                cam_gen = GradCAMGenerator(model, target_layer_name="layer4")
                heatmap = cam_gen.generate(tensor)
                gradcam_image = create_gradcam_overlay(image, heatmap, alpha=0.45)
                cam_gen.cleanup()

                col_orig, col_cam = st.columns(2)
                with col_orig:
                    st.image(image, caption="Original X-Ray", use_container_width=True)
                with col_cam:
                    st.image(
                        gradcam_image,
                        caption="Grad-CAM Activation Map",
                        use_container_width=True,
                    )

                st.markdown(
                    """
                    <div class="info-box">
                        <strong>What is Grad-CAM?</strong><br>
                        Gradient-weighted Class Activation Mapping highlights the regions
                        of the X-ray that the AI model focused on most when making its
                        prediction. Warmer colors (red/yellow) indicate higher attention.
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

            except Exception as e:
                st.warning(f"⚠️ Grad-CAM could not be generated: {e}")
                logger.warning(f"Grad-CAM error: {e}", exc_info=True)

            custom_divider()

            # ---- PDF REPORT ----
            section_header("📄", "Download Report")

            try:
                pdf_bytes = generate_pdf_report(
                    original_image=image,
                    gradcam_image=gradcam_image,
                    label=label,
                    confidence=confidence,
                    risk_level=risk_level,
                    probabilities=prob_dict,
                )
                st.download_button(
                    label="📥 Download PDF Report",
                    data=pdf_bytes,
                    file_name=f"xray_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf",
                    mime="application/pdf",
                    use_container_width=True,
                )
            except Exception as e:
                st.warning(f"⚠️ PDF generation failed: {e}")
                logger.warning(f"PDF error: {e}", exc_info=True)

            # ---- SAVE TO HISTORY ----
            st.session_state.history.append(
                {
                    "Timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "File": uploaded_file.name,
                    "Prediction": label,
                    "Confidence": f"{confidence * 100:.1f}%",
                    "Risk Level": risk_level,
                }
            )

            custom_divider()

            # Disclaimer
            st.markdown(
                """
                <div class="info-box" style="border-left-color: #FF9100;">
                    ⚠️ <strong>Medical Disclaimer:</strong> This tool is designed for
                    research and educational purposes only. It is NOT a substitute for
                    professional medical advice, diagnosis, or treatment. Always consult
                    a qualified healthcare provider for clinical decisions.
                </div>
                """,
                unsafe_allow_html=True,
            )


# -------------------------------------------------------------------
#  HISTORY PAGE
# -------------------------------------------------------------------
elif page == "📜 History":
    st.markdown(
        """
        <div class="animate-in">
            <h1 style="font-size: 1.8rem; font-weight: 700;
                       background: linear-gradient(135deg, #0A7E8C, #0EA5B5);
                       -webkit-background-clip: text; -webkit-text-fill-color: transparent;
                       background-clip: text; margin-bottom: 0.2rem;">
                Prediction History
            </h1>
            <p style="color: #9AA0A6; font-size: 0.9rem;">
                Review all predictions made during this session.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    custom_divider()

    if st.session_state.history:
        import pandas as pd

        df = pd.DataFrame(st.session_state.history)

        # Summary cards
        total = len(df)
        pneumonia_count = len(df[df["Prediction"] == "PNEUMONIA"])
        normal_count = total - pneumonia_count

        col1, col2, col3 = st.columns(3)
        with col1:
            metric_card("📊", "Total Predictions", str(total), "dataset")
        with col2:
            metric_card("🟢", "Normal", str(normal_count), "accuracy")
        with col3:
            metric_card("🔴", "Pneumonia", str(pneumonia_count), "f1")

        custom_divider()

        # Table
        st.dataframe(
            df,
            use_container_width=True,
            hide_index=True,
            column_config={
                "Timestamp": st.column_config.TextColumn("🕐 Timestamp"),
                "File": st.column_config.TextColumn("📄 File"),
                "Prediction": st.column_config.TextColumn("🏷️ Prediction"),
                "Confidence": st.column_config.TextColumn("📊 Confidence"),
                "Risk Level": st.column_config.TextColumn("⚠️ Risk"),
            },
        )

        # Export
        csv_data = df.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="📥 Export as CSV",
            data=csv_data,
            file_name=f"prediction_history_{datetime.now().strftime('%Y%m%d')}.csv",
            mime="text/csv",
            use_container_width=True,
        )

        # Clear history
        if st.button("🗑️ Clear History", use_container_width=True):
            st.session_state.history = []
            st.rerun()
    else:
        st.markdown(
            """
            <div style="text-align: center; padding: 3rem; color: #5F6368;">
                <div style="font-size: 3rem; margin-bottom: 1rem;">📭</div>
                <div style="font-size: 1.1rem; font-weight: 500;">No predictions yet</div>
                <div style="font-size: 0.85rem; margin-top: 0.5rem;">
                    Navigate to the <strong>Predict</strong> page and upload a chest X-ray
                    to get started.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )


# -------------------------------------------------------------------
#  ABOUT PAGE
# -------------------------------------------------------------------
elif page == "ℹ️ About":
    st.markdown(
        """
        <div class="animate-in">
            <h1 style="font-size: 1.8rem; font-weight: 700;
                       background: linear-gradient(135deg, #0A7E8C, #0EA5B5);
                       -webkit-background-clip: text; -webkit-text-fill-color: transparent;
                       background-clip: text; margin-bottom: 0.2rem;">
                About This Project
            </h1>
        </div>
        """,
        unsafe_allow_html=True,
    )

    custom_divider()

    st.markdown(
        """
        <div class="info-box">
            <strong>🫁 Chest X-Ray Pneumonia Detection System</strong><br><br>
            This application uses a deep learning model based on <strong>ResNet-18</strong>
            (transfer learning from ImageNet) to classify chest X-ray images as either
            <span style="color: #00E676; font-weight: 600;">NORMAL</span> or
            <span style="color: #FF1744; font-weight: 600;">PNEUMONIA</span>.<br><br>
            The model was trained on the
            <a href="https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia"
               style="color: #0EA5B5;" target="_blank">
                Kaggle Chest X-Ray Pneumonia Dataset
            </a>
            with class-weighted loss, data augmentation, learning rate scheduling,
            and early stopping to achieve robust performance.
        </div>
        """,
        unsafe_allow_html=True,
    )

    custom_divider()
    section_header("🛠️", "Technology Stack")

    col1, col2 = st.columns(2)
    with col1:
        st.markdown(
            """
            <div class="info-box">
                <strong>Backend / ML</strong><br>
                • PyTorch 2.x<br>
                • TorchVision (ResNet-18)<br>
                • Grad-CAM (custom hooks)<br>
                • scikit-learn (metrics)<br>
                • NumPy / Pillow
            </div>
            """,
            unsafe_allow_html=True,
        )
    with col2:
        st.markdown(
            """
            <div class="info-box">
                <strong>Frontend / UI</strong><br>
                • Streamlit (wide layout)<br>
                • Plotly (interactive charts)<br>
                • Custom CSS (glassmorphism)<br>
                • fpdf2 (PDF reports)<br>
                • Google Fonts (Inter)
            </div>
            """,
            unsafe_allow_html=True,
        )

    custom_divider()
    section_header("📊", "Key Features")

    features = [
        ("🔬", "Real-time Inference", "Upload and analyze chest X-rays in seconds"),
        ("🔍", "Grad-CAM", "Visual explanations of model attention regions"),
        ("📊", "Interactive Charts", "Plotly-powered probability visualizations"),
        ("📄", "PDF Reports", "Downloadable clinical-style analysis reports"),
        ("📜", "Prediction History", "Track and export all session predictions"),
        ("🎨", "Dark Mode UI", "Professional healthcare-themed interface"),
    ]

    cols = st.columns(3)
    for i, (icon, title, desc) in enumerate(features):
        with cols[i % 3]:
            st.markdown(
                f"""
                <div class="metric-card" style="text-align: center; min-height: 130px;">
                    <div style="font-size: 2rem; margin-bottom: 0.5rem;">{icon}</div>
                    <div style="font-weight: 700; color: #E8EAED; font-size: 0.95rem;
                                margin-bottom: 0.3rem;">{title}</div>
                    <div style="color: #9AA0A6; font-size: 0.78rem;">{desc}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    custom_divider()

    st.markdown(
        """
        <div style="text-align: center; padding: 1.5rem; color: #5F6368; font-size: 0.8rem;">
            Made with ❤️ using PyTorch &amp; Streamlit<br>
            <span style="font-size: 0.7rem;">
                © 2024 Chest X-Ray Pneumonia Detection Project
            </span>
        </div>
        """,
        unsafe_allow_html=True,
    )
