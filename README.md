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

Verified on the standard **Kermany test split (624 images: 234 NORMAL, 390 PNEUMONIA)**. These images are never used for training or validation.
Reproduce with `python scripts/evaluate_model.py --parquet <test.parquet>` (writes `reports/evaluation.json` and the plots below).

| Metric | Value |
|---|---|
| Accuracy | **84.8%** |
| Macro F1 | 0.841 |
| ROC-AUC | 0.916 |
| PR-AUC (pneumonia) | 0.934 |
| Pneumonia recall (sensitivity) | 84.6% |
| Normal recall (specificity) | 85.0% |
| Pneumonia precision | 90.4% |

> **Correction:** earlier versions of this README reported 92.79% accuracy / 0.9923 ROC-AUC. Those numbers could not be reproduced with the committed `xray_model.pth` on the standard test split, so they were replaced with the verified results above.

### Decision-threshold trade-off

For screening, missing a pneumonia case is usually costlier than a false alarm. The model outputs a probability, and the threshold controls this trade-off:

| Threshold | Sensitivity | Specificity | Missed pneumonia | False alarms |
|---|---|---|---|---|
| 0.3 | 93.8% | 71.4% | 24 | 67 |
| 0.4 | 90.5% | 79.5% | 37 | 48 |
| **0.5 (default)** | 84.6% | 85.0% | 60 | 35 |
| 0.6 | 79.5% | 89.7% | 80 | 24 |

These rows are computed on the test split, so they describe the trade-off rather than a tuned threshold. A tuned threshold should be picked on a separate validation set.

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
