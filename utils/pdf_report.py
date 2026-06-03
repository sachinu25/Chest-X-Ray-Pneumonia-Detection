"""
PDF report generation for the Streamlit dashboard using Matplotlib.

Bypasses external PDF dependencies by using Matplotlib's native PDF export backend.
Creates a professional, clean one-page clinical-style report.
"""

import io
from datetime import datetime
from typing import Dict, Optional

import matplotlib
matplotlib.use("Agg")  # Non-interactive backend
import matplotlib.pyplot as plt
from PIL import Image


def generate_pdf_report(
    original_image: Image.Image,
    gradcam_image: Optional[Image.Image],
    label: str,
    confidence: float,
    risk_level: str,
    probabilities: Dict[str, float],
) -> bytes:
    """
    Generate a professional PDF report using Matplotlib's PDF backend.

    Args:
        original_image: The uploaded X-ray image.
        gradcam_image: The Grad-CAM overlay image.
        label: Predicted class label.
        confidence: Prediction confidence (0-1).
        risk_level: Risk level (e.g., "HIGH RISK").
        probabilities: Dict mapping class names to probabilities.

    Returns:
        PDF file content as bytes.
    """
    # Use dark/light colors consistent with our theme
    primary_color = "#0A7E8C"
    text_color = "#1E1E1E"
    normal_color = "#00C853"
    pneumonia_color = "#FF1744"

    # Create figure
    fig = plt.figure(figsize=(8.5, 11), dpi=150)

    # 1. Header Area (draw as a rectangle on a background ax or using figure text)
    # Define grid: header at top, images in middle, stats at bottom
    # We will use simple relative positioning coordinates in inches/fractions
    
    # Background decoration
    fig.patch.set_facecolor("#FAFAFA")

    # Header title text
    fig.text(0.5, 0.94, "Chest X-Ray Analysis Report", 
             fontsize=22, fontweight="bold", color="white", 
             ha="center", va="center",
             bbox=dict(boxstyle="square,pad=0.6", facecolor=primary_color, edgecolor="none"))
    
    fig.text(0.5, 0.89, f"Report Generated: {datetime.now().strftime('%B %d, %Y at %I:%M %p')}", 
             fontsize=10, fontstyle="italic", color="#555555", ha="center")

    # 2. Main Panel for Images
    # We place original image on the left, Grad-CAM on the right
    ax_orig = fig.add_axes([0.1, 0.45, 0.38, 0.38])
    ax_orig.imshow(original_image)
    ax_orig.set_title("Uploaded Chest X-Ray", fontsize=11, fontweight="bold", pad=8)
    ax_orig.axis("off")

    if gradcam_image is not None:
        ax_cam = fig.add_axes([0.52, 0.45, 0.38, 0.38])
        ax_cam.imshow(gradcam_image)
        ax_cam.set_title("Grad-CAM Attention Map", fontsize=11, fontweight="bold", pad=8)
        ax_cam.axis("off")
    else:
        # Placeholder or empty axes if no Grad-CAM
        ax_empty = fig.add_axes([0.52, 0.45, 0.38, 0.38])
        ax_empty.text(0.5, 0.5, "Grad-CAM explanation\nnot available", 
                      ha="center", va="center", color="#888888")
        ax_empty.axis("off")

    # Draw divider line
    line_y = 0.41
    fig.add_axes([0.1, line_y, 0.8, 0.001]).axis("off")
    plt.plot([0.1, 0.9], [line_y, line_y], color="#D0D0D0", transform=fig.transFigure, linewidth=1.5)

    # 3. Clinical Results Panel
    # Put text info on the left, horizontal bar chart on the right
    ax_text = fig.add_axes([0.1, 0.16, 0.4, 0.22])
    ax_text.axis("off")

    pred_color = pneumonia_color if label == "PNEUMONIA" else normal_color

    result_text = (
        f"AI DIAGNOSTIC RESULTS\n"
        f"--------------------------------------------------\n\n"
        f"Predicted Class:  "
    )
    
    ax_text.text(0, 0.95, "AI DIAGNOSTIC RESULTS", fontsize=12, fontweight="bold", color=primary_color)
    ax_text.text(0, 0.88, "-------------------------------------------------------------", color="#C0C0C0")
    
    ax_text.text(0, 0.70, "Predicted Class:", fontsize=11, color="#333333")
    ax_text.text(0.35, 0.70, label, fontsize=12, fontweight="bold", color=pred_color)

    ax_text.text(0, 0.50, "Confidence:", fontsize=11, color="#333333")
    ax_text.text(0.35, 0.50, f"{confidence * 100:.2f}%", fontsize=12, fontweight="bold", color="#1E1E1E")

    ax_text.text(0, 0.30, "Risk Level:", fontsize=11, color="#333333")
    ax_text.text(0.35, 0.30, risk_level, fontsize=12, fontweight="bold", color=pred_color)

    # Probabilities bar chart on the right
    ax_chart = fig.add_axes([0.55, 0.18, 0.35, 0.16])
    classes = list(probabilities.keys())
    probs = [probabilities[cls] * 100 for cls in classes]
    colors = [normal_color if cls == "NORMAL" else pneumonia_color for cls in classes]

    bars = ax_chart.barh(classes, probs, color=colors, height=0.45, edgecolor="none")
    ax_chart.set_xlim(0, 110)
    ax_chart.set_xlabel("Probability (%)", fontsize=9, fontweight="bold")
    ax_chart.spines["top"].set_visible(False)
    ax_chart.spines["right"].set_visible(False)
    ax_chart.spines["left"].set_color("#C0C0C0")
    ax_chart.spines["bottom"].set_color("#C0C0C0")
    ax_chart.tick_params(axis="both", colors="#555555", labelsize=9)

    # Add text labels on the bars
    for bar in bars:
        width = bar.get_width()
        ax_chart.text(width + 2, bar.get_y() + bar.get_height()/2, 
                      f"{width:.1f}%", 
                      va="center", ha="left", fontsize=9, fontweight="bold", color="#333333")

    # 4. Footer & Medical Disclaimer
    disclaimer_text = (
        "MEDICAL DISCLAIMER: This report is generated automatically by an AI-assisted diagnostic software tool "
        "and is intended for informational/educational support only. It does NOT replace a clinical diagnosis or "
        "examination by a qualified health professional. Treatment decisions should not be based solely on this report."
    )
    fig.text(0.5, 0.06, disclaimer_text, 
             fontsize=7.5, color="#777777", ha="center", va="center", wrap=True,
             bbox=dict(boxstyle="round,pad=0.5", facecolor="#F0F0F0", edgecolor="#E0E0E0", alpha=0.8))

    # Save to BytesIO buffer in PDF format
    buf = io.BytesIO()
    fig.savefig(buf, format="pdf", facecolor=fig.get_facecolor(), bbox_inches="tight")
    plt.close(fig)
    
    return buf.getvalue()
