# 🫁 Chest X-Ray Pneumonia Detection System

<div align="center">

### AI-Powered Medical Image Classification Using Deep Learning

![Python](https://img.shields.io/badge/Python-3.10+-blue?style=for-the-badge&logo=python)
![PyTorch](https://img.shields.io/badge/PyTorch-Deep%20Learning-red?style=for-the-badge&logo=pytorch)
![FastAPI](https://img.shields.io/badge/FastAPI-REST%20API-green?style=for-the-badge&logo=fastapi)
![Docker](https://img.shields.io/badge/Docker-Containerized-blue?style=for-the-badge&logo=docker)
![AWS](https://img.shields.io/badge/AWS-Cloud-orange?style=for-the-badge&logo=amazonaws)

</div>

---

# 📖 Overview

This project is an end-to-end AI-powered medical image analysis system designed to detect **Pneumonia** from Chest X-Ray images using **Transfer Learning with ResNet18**.

The system provides:

- Deep Learning-based diagnosis
- FastAPI REST API
- Dockerized deployment
- Explainable AI (Grad-CAM)
- Automated ML pipeline
- AWS deployment readiness

---

# 🎯 Problem Statement

Pneumonia is a severe respiratory infection affecting millions of people worldwide.

Manual diagnosis through chest radiographs:

- Requires expert radiologists
- Time consuming
- Subject to human variability
- Difficult in resource-constrained areas

This project aims to assist healthcare professionals by providing rapid AI-assisted screening.

---

# 🚀 Features

| Feature | Status |
|----------|----------|
| Pneumonia Detection | ✅ |
| ResNet18 Transfer Learning | ✅ |
| FastAPI Backend | ✅ |
| Docker Deployment | ✅ |
| Grad-CAM Explainability | ✅ |
| PDF Report Generation | ✅ |
| CI/CD Pipeline | ✅ |
| AWS Deployment Support | ✅ |

---

# 🏗 System Architecture

```text
                User
                  │
                  ▼
        Upload Chest X-Ray
                  │
                  ▼
              FastAPI
                  │
                  ▼
         Image Processing
                  │
                  ▼
          ResNet18 Model
                  │
                  ▼
          Prediction
                  │
                  ▼
       Confidence Score
```

---

# 🔬 Machine Learning Pipeline

```text
Data Collection
      │
      ▼
Data Validation
      │
      ▼
Data Transformation
      │
      ▼
Image Augmentation
      │
      ▼
Model Training
      │
      ▼
Evaluation
      │
      ▼
Model Deployment
```

---

# 🏆 Model Performance

Verified on the standard **Kermany test split (624 images: 234 NORMAL, 390 PNEUMONIA)**. These images are never used for training, model selection or threshold choice.
Reproduce with `python scripts/evaluate_model.py --parquet <test.parquet>` (writes `reports/evaluation.json` and the plots below).

| Metric | v1 (original model, thr 0.5) | **v2 (current, thr 0.835)** |
|---|---|---|
| Accuracy | 84.8% | **87.7%** |
| Pneumonia recall (sensitivity) | 84.6% | **99.0%** (4 of 390 missed) |
| Normal recall (specificity) | **85.0%** | 68.8% |
| Pneumonia precision | **90.4%** | 84.1% |
| Macro F1 | 0.841 | **0.858** |
| ROC-AUC | 0.916 | 0.918 |
| PR-AUC (pneumonia) | **0.934** | 0.932 |

**What changed in v2** (`scripts/train_v2.py`):
- **Patient-level validation split.** The official val split has only 16 images, so 15% of patients are held out instead. No patient appears in both train and validation.
- **Histogram equalisation** at training and inference, plus stronger brightness/contrast augmentation. Kermany's train and test images differ in intensity, which caused many normal test images to be flagged as pneumonia.
- **Training recipe:** AdamW + one-cycle LR, bf16 mixed precision, class-weighted loss with label smoothing, horizontal-flip test-time augmentation.
- **Model selection:** ResNet18 and EfficientNet-B0 were compared on validation ROC-AUC, and ResNet18 won. The decision threshold (0.835) maximises balanced accuracy on validation, not on test.

**Honest trade-off:** v2 catches almost every pneumonia case (99% recall), but raises more false alarms on normal X-rays. ROC-AUC is essentially unchanged. Validation ROC-AUC is about 0.998 versus 0.918 on test, which shows a real **distribution shift** between the Kermany train and test images. This is the main limit on further gains.

> **Correction:** earlier versions of this README reported 92.79% accuracy / 0.9923 ROC-AUC. Those numbers could not be reproduced with the committed model on the standard test split.

---

# 📊 Model Performance Visualizations

<p align="center">
  <img src="assets/confusion_matrix.png" width="380">
  <img src="assets/roc_curve.png" width="380">
  <img src="assets/pr_curve.png" width="380">
</p>

## 📈 Training vs Validation Accuracy

<p align="center">
  <img src="assets/training_validation_accuracy.png" width="850">
</p>

---
# 📂 Project Structure

```bash
.
├── app.py
├── streamlit_app.py
├── train.py
├── evaluate.py
├── Dockerfile
├── requirements.txt
│
├── utils/
│   ├── model_utils.py
│   ├── gradcam.py
│   ├── pdf_report.py
│   └── logger.py
│
├── xray/
│   ├── components/
│   ├── pipeline/
│   ├── ml/
│   ├── entity/
│   └── constant/
│
├── scripts/
│
└── .github/
```

---

# 🧠 Model Architecture

### Backbone

- ResNet18 (ImageNet Pretrained)

### Classification Head

```python
Dropout
↓
Linear(512 → 256)
↓
ReLU
↓
Dropout
↓
Linear(256 → 2)
```

### Classes

| Label | Class |
|---------|---------|
| 0 | NORMAL |
| 1 | PNEUMONIA |

---

# ⚙️ Installation

## Clone Repository

```bash
git clone https://github.com/sachinu25/Chest-X-Ray-Pneumonia-Detection.git
cd Chest-X-Ray-Pneumonia-Detection
```

## Create Virtual Environment

```bash
python -m venv venv
```

### Windows

```bash
venv\Scripts\activate
```

### Linux/Mac

```bash
source venv/bin/activate
```

## Install Dependencies

```bash
pip install -r requirements.txt
```

---

# 🏋️ Training

```bash
python train.py
```

---

# 📊 Evaluation

```bash
python evaluate.py
```

---

# 🚀 Run FastAPI

```bash
uvicorn app:app --host 0.0.0.0 --port 8000
```

Swagger UI:

```text
http://localhost:8000/docs
```

---

# 🌐 API Endpoints

## Health Check

```http
GET /health
```

### Response

```json
{
  "status": "healthy",
  "model_loaded": true,
  "device": "cuda"
}
```

---

## Prediction Endpoint

```http
POST /predict
```

### Response

```json
{
  "prediction_label": "PNEUMONIA",
  "confidence": 0.97
}
```

---

# 🐳 Docker Deployment

## Build Image

```bash
docker build -t xray-classifier .
```

## Run Container

```bash
docker run -p 8000:8000 xray-classifier
```

---

# ☁️ AWS Deployment

Supported Platforms:

- AWS EC2
- AWS ECS
- AWS ECR
- Elastic Beanstalk
- Application Load Balancer

---

# 🛠 Tech Stack

| Category | Technology |
|------------|------------|
| Language | Python |
| Deep Learning | PyTorch |
| Model | ResNet18 |
| Backend | FastAPI |
| UI | Streamlit |
| Containerization | Docker |
| Cloud | AWS |
| CI/CD | GitHub Actions |

---

# 📈 Future Improvements

- [ ] Multi-class lung disease classification
- [ ] ONNX optimization
- [ ] TensorRT acceleration
- [ ] Kubernetes deployment
- [ ] Model monitoring
- [ ] Explainable AI dashboard

---

# ⚠️ Disclaimer

This project is intended for educational and research purposes only.

It should not be used as a replacement for professional medical diagnosis.

---

# 👨‍💻 Author

**Sachin Upadhyay**

AI Engineer | Machine Learning Enthusiast | Data Science Aspirant

---

# ⭐ If you found this project useful

Give it a star ⭐ and support the repository.
