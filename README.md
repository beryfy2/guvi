# 🛡️ AI Security & Fraud Detection Dashboard

A Flask-based web application that provides a single dashboard to run multiple
machine learning and deep learning models for security and fraud detection.

---

## 🚀 Features

- Phishing Detection (BERT)
- Fake Image Detection (CNN)
- AI Image Detection (ViT)
- Fake Audio Detection (MFCC-13 & MFCC-40)
- Credit Card Fraud Detection (CSV upload)
- Confidence scores
- Bar chart summary for fraud results
- Fraud rows displayed in table

---

## 🧰 Requirements

- Python **3.10.19**
- Conda (recommended)

---

## 🐍 Create Environment

```bash
conda create -n venv python=3.10.19
conda activate venv
```

## 📦 Install Dependencies

```bash
pip install -r requirements.txt
```

### or

```bash
pip install flask==2.3.3 werkzeug==2.3.7 tensorflow==2.13.0 torch torchvision torchaudio transformers scikit-learn pandas numpy librosa opencv-python pillow joblib tqdm
```

##  Run Application

```bash
python app.py
```

## 📂 Project Structure

```bash
Guvi/
├── app.py
├── model_loader.py
├── requirements.txt
├── model/
├── templates/
│   └── index.html
├── static/
│   ├── styles.css
│   └── app.js
└── uploads/
```
