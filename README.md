# 🫁 X-ray Lung Disease Classifier

> Deep learning–based pneumonia detection from chest X-ray images using transfer learning (ResNet18).

![Python](https://img.shields.io/badge/Python-3.10+-blue?logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red?logo=pytorch&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-REST%20API-green?logo=fastapi&logoColor=white)
![Docker](https://img.shields.io/badge/Docker-Containerized-blue?logo=docker&logoColor=white)
![AWS](https://img.shields.io/badge/AWS-Cloud%20Deployment-orange?logo=amazonaws&logoColor=white)
![CI/CD](https://img.shields.io/badge/CI%2FCD-GitHub%20Actions-black?logo=githubactions&logoColor=white)

---

## Problem Statement

Pneumonia is a lung infection that can be life-threatening if not detected early. Chest X-ray imaging is one of the most widely used diagnostic tools. This project builds an end-to-end deep learning API that classifies chest X-ray images as **Pneumonia** or **Normal**.

## Solution

| Feature | Details |
|---------|---------|
| **Model** | ResNet18 (transfer learning, ImageNet pre-trained) |
| **Framework** | PyTorch |
| **API** | FastAPI (REST) + Streamlit (interactive UI) |
| **Training** | Early stopping, LR scheduling, gradient clipping |
| **Evaluation** | Accuracy, Precision, Recall, F1-Score, Confusion Matrix |
| **Deployment** | Docker + AWS ECR + GitHub Actions CI/CD |

## Model Architecture

- **Backbone**: ResNet18 pre-trained on ImageNet (11.2M parameters)
- **Classification Head**: Dropout → FC(512→256) → ReLU → Dropout → FC(256→2)
- **Loss Function**: CrossEntropyLoss (raw logit output)
- **Optimizer**: SGD with momentum + weight decay
- **Scheduler**: StepLR with early stopping (patience=5)

## Project Structure

```
├── app.py                          # FastAPI REST API
├── streamlit_app.py                # Streamlit interactive UI
├── train.py                        # Training entry point
├── evaluate.py                     # Standalone model evaluation script
├── Dockerfile                      # Multi-stage production Docker build
├── requirements.txt                # Pinned dependencies
│
├── xray/                           # Main package
│   ├── components/                 # Pipeline components
│   │   ├── data_ingestion.py       #   Local data validation and extraction
│   │   ├── data_transformation.py  #   Augmentation, normalization, DataLoaders
│   │   ├── model_training.py       #   Training loop with early stopping
│   │   └── model_evaluation.py     #   Comprehensive metrics
│   │
│   ├── constant/
│   │   └── training_pipeline/
│   │       └── __init__.py         # All hyperparameters & constants
│   │
│   ├── entity/
│   │   ├── config_entity.py        # Stage configurations
│   │   └── artifacts_entity.py     # Stage output contracts
│   │
│   ├── ml/model/
│   │   └── arch.py                 # XRayClassifier (ResNet-18)
│   │
│   ├── pipeline/
│   │   └── train_pipeline.py       # End-to-end orchestrator
│   │
│   ├── exception.py                # Custom exception with traceback
│   └── logger.py                   # File + console logging
│
├── scripts/
│   └── start_up.sh                 # EC2 provisioning script
│
└── .github/workflows/
    └── main.yml                    # CI/CD pipeline
```

## Quick Start

### 1. Clone & Setup

```bash
git clone <repo-url>
cd lung-disease-diagnosis-main

# Create virtual environment
python -m venv venv
source venv/bin/activate   # Linux/Mac
venv\Scripts\activate      # Windows

# Install dependencies
pip install -r requirements.txt
```

### 2. Set Environment Variables

```bash
export AWS_ACCESS_KEY_ID=<your-key>
export AWS_SECRET_ACCESS_KEY=<your-secret>
export AWS_DEFAULT_REGION=us-east-1
export AWS_ACCOUNT_ID=<your-account-id>
```

### 3. Train the Model

```bash
python train.py
```

### 4. Run the API Server

```bash
uvicorn app:app --host 0.0.0.0 --port 8000
```

Then visit `http://localhost:8000/docs` for interactive API documentation.

### 5. Run Streamlit UI

```bash
streamlit run streamlit_app.py
```

### 6. Docker Deployment

```bash
docker build -t xray-classifier .
docker run -p 8000:8000 xray-classifier
```

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/` | Health check |
| `GET` | `/health` | Detailed health status |
| `POST` | `/predict` | Upload X-ray image → get prediction |

### Example Response

```json
{
  "prediction_index": 1,
  "prediction_label": "PNEUMONIA",
  "confidence": 0.9723,
  "probabilities": {
    "NORMAL": 0.0277,
    "PNEUMONIA": 0.9723
  }
}
```

## Tech Stack

| Component | Technology |
|-----------|------------|
| Language | Python 3.10+ |
| Deep Learning | PyTorch 2.0+ |
| Model | ResNet18 (Transfer Learning) |
| API | FastAPI |
| Interactive UI | Streamlit |
| Containerization | Docker |
| Cloud | AWS (ECR, EC2, App Runner) |
| CI/CD | GitHub Actions |

## Key Design Decisions

1. **Transfer Learning over Custom CNN**: ResNet18 pre-trained on ImageNet provides dramatically better feature extraction than a small custom CNN (~17K params vs 11M params)
2. **CrossEntropyLoss with Raw Logits**: Numerically stable; avoids the sigmoid/softmax + loss mismatch bug
3. **Validation Split from Training Data**: Prevents data leakage — test set is never seen during training
4. **Early Stopping**: Prevents overfitting by monitoring validation loss
5. **Matching Inference Transforms**: Inference pipeline applies the exact same normalization as training

## Disclaimer

⚠️ This is a **decision-support tool** for research purposes only. It is **not** a substitute for professional medical diagnosis. Always consult a qualified healthcare provider.

## License

This project is for educational and research purposes.
