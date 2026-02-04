# =========================
# COMMON IMPORTS
# =========================
import numpy as np
from PIL import Image, ImageChops, ImageEnhance
import joblib
import librosa
import cv2
import torch
from torchvision.transforms import Compose, Resize, ToTensor, Normalize
from transformers import ViTForImageClassification, ViTImageProcessor
from transformers import BertTokenizer, BertForSequenceClassification
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.vgg16 import preprocess_input
from tensorflow.keras.preprocessing.sequence import pad_sequences

# =========================
# DEVICE
# =========================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# =========================
# LOAD MODELS
# =========================

print("Loading ViT AI Image...")
vit_model = ViTForImageClassification.from_pretrained(
    "model/fakeImage_model/aiImage_model"
)
vit_processor = ViTImageProcessor.from_pretrained(
    "model/fakeImage_model/aiImage_model"
)
vit_model.to(DEVICE)
vit_model.eval()
vit_labels = vit_model.config.id2label

print("Loading CNN Fake Image...")
cnn_image_model = load_model(
    "model/fakeImage_model/image_model.keras",
    compile=False
)

print("Loading Audio Model V1...")
audio_model_v1 = load_model(
    "model/fakeAudio_Model/voice_detection_model.keras"
)

print("Loading Audio Model V2...")
audio_model_v2 = load_model(
    "model/fakeAudio_model/FakeAudioDetectionModel.keras"
)

print("Loading Card Fraud...")
card_model = joblib.load(
    "model/cardFraud_model/credit_card_model.pkl"
)

print("Loading Phishing BERT...")
phishing_model = BertForSequenceClassification.from_pretrained(
    "model/phising_model"
)
phishing_tokenizer = BertTokenizer.from_pretrained(
    "model/phising_tokenizer"
)
phishing_model.to(DEVICE)
phishing_model.eval()

print("ALL MODELS LOADED ✅")

# =========================
# AI IMAGE (ViT)
# =========================

vit_transform = Compose([
    Resize((vit_processor.size["height"],
            vit_processor.size["height"])),
    ToTensor(),
    Normalize(
        mean=vit_processor.image_mean,
        std=vit_processor.image_std
    )
])

def predict_ai_image(path):
    img = Image.open(path).convert("RGB")
    tensor = vit_transform(img).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        out = vit_model(pixel_values=tensor)
        probs = torch.softmax(out.logits, dim=1)
        pid = torch.argmax(probs, dim=1).item()
        conf = probs[0][pid].item()

    return vit_labels[pid], conf

# =========================
# CNN FAKE IMAGE
# =========================

def predict_fake_image(path):
    img = cv2.imread(path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img,(224,224))
    img = preprocess_input(img)
    img = img.reshape(1,224,224,3)

    pred = cnn_image_model.predict(img)[0][0]
    return ("REAL IMAGE",pred) if pred>0.5 else ("FAKE IMAGE",1-pred)

# =========================
# AUDIO V1 (MFCC 13)
# =========================

def predict_fake_audio_v1(path, max_length=500):

    audio, sr = librosa.load(path, sr=None)

    mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=13)
    mfccs = mfccs.T

    padded = pad_sequences(
        [mfccs],
        maxlen=max_length,
        dtype="float32",
        padding="post",
        truncating="post"
    )

    padded = padded[..., np.newaxis]

    prediction = audio_model_v1.predict(padded)

    predicted_class = prediction.argmax(axis=1)[0]

    if predicted_class == 1:
        return "FAKE AUDIO", float(np.max(prediction))
    else:
        return "REAL AUDIO", float(np.max(prediction))

# =========================
# AUDIO V2 (MFCC 40)
# =========================

def predict_fake_audio_v2(path,max_len=500):

    audio,_ = librosa.load(path,sr=16000)
    mfcc = librosa.feature.mfcc(y=audio,sr=16000,n_mfcc=40)

    if mfcc.shape[1]<max_len:
        mfcc = np.pad(mfcc,((0,0),(0,max_len-mfcc.shape[1])))
    else:
        mfcc = mfcc[:,:max_len]

    mfcc = mfcc[...,np.newaxis]
    mfcc = np.expand_dims(mfcc,0)

    pred = audio_model_v2.predict(mfcc)[0][0]

    return ("FAKE AUDIO",pred) if pred>0.5 else ("REAL AUDIO",1-pred)

# =========================
# CARD FRAUD CSV
# =========================

def predict_card_fraud(path):

    import pandas as pd

    df = pd.read_csv(path)

    # Drop target column if present
    if "Class" in df.columns:
        df = df.drop(columns=["Class"])

    # Drop time column if present
    if "Time" in df.columns:
        df = df.drop(columns=["Time"])

    preds = card_model.predict(df)

    fraud_count = int((preds == 1).sum())
    normal_count = int((preds == 0).sum())

    return f"Fraud: {fraud_count} | Normal: {normal_count}", 1.0


# =========================
# PHISHING BERT
# =========================

def predict_phishing(text):

    inputs = phishing_tokenizer(
        text,return_tensors="pt",
        truncation=True,padding=True,
        max_length=512
    )

    inputs = {k:v.to(DEVICE) for k,v in inputs.items()}

    with torch.no_grad():
        out = phishing_model(**inputs)

    probs = torch.softmax(out.logits,dim=1)[0]
    pid = torch.argmax(probs).item()
    conf = probs[pid].item()

    return ("PHISHING",conf) if pid==1 else ("LEGITIMATE",conf)
