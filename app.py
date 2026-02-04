from pathlib import Path
import uuid
from flask import Flask,render_template,request
from werkzeug.utils import secure_filename

from model_loader import (
    predict_ai_image,
    predict_fake_image,
    predict_fake_audio_v1,
    predict_fake_audio_v2,
    predict_card_fraud,
    predict_phishing
)

# =========================
BASE = Path(__file__).parent
UPLOADS = BASE/"uploads"
UPLOADS.mkdir(exist_ok=True)

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"]=200*1024*1024

# =========================
MODEL_CATALOG = [
    {"id":"phishing","name":"Phishing Message Detection","input":"text"},
    {"id":"fake_image","name":"Fake / Manipulated Image Detection","input":"image"},
    {"id":"ai_image","name":"AI-Generated Image Detection","input":"image"},
    {"id":"fake_audio_v2","name":"Fake Voice / Audio Detection","input":"audio"},
    {"id":"fake_audio_v1","name":"Fake Voice Detection (Beta)","input":"audio"},
    {"id":"credit_card_fraud","name":"Credit Card Fraud Detection","input":"tabular"}
]

# =========================

def save_file(f):
    if not f or not f.filename:
        return None
    name = secure_filename(f.filename)
    name = f"{Path(name).stem}-{uuid.uuid4().hex[:8]}{Path(name).suffix}"
    path = UPLOADS/name
    f.save(path)
    return path

# =========================

def run_inference(mid,text,file):

    if mid=="phishing":
        return predict_phishing(text)

    if mid=="fake_image":
        return predict_fake_image(file)

    if mid=="ai_image":
        return predict_ai_image(file)

    if mid=="fake_audio_v1":
        return predict_fake_audio_v1(file)

    if mid=="fake_audio_v2":
        return predict_fake_audio_v2(file)

    if mid=="credit_card_fraud":
        return predict_card_fraud(file)

    return "Unknown",0

# =========================

@app.route("/")
def index():
    return render_template(
        "index.html",
        models=MODEL_CATALOG,
        selected_id=MODEL_CATALOG[0]["id"],
        result=None,
        text_value=""
    )

@app.route("/predict",methods=["POST"])
def predict():

    mid = request.form.get("model")
    text = request.form.get("text","")
    file = request.files.get("file")
    path = save_file(file)

    label,conf = run_inference(mid,text,path)

    result = {
        "verdict":label,
        "note":"Inference Completed",
        "details":[
            {"label":"Confidence","value":f"{conf:.3f}"}
        ]
    }

    return render_template(
        "index.html",
        models=MODEL_CATALOG,
        selected_id=mid,
        result=result,
        text_value=text
    )

if __name__=="__main__":
    app.run(debug=True)
