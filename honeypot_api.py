from flask import Flask, request, jsonify
import torch, re, requests
from transformers import (
    BertTokenizer,
    BertForSequenceClassification,
    AutoTokenizer,
    AutoModelForCausalLM
)

# ============================
# CONFIG
# ============================
import os
API_KEY = os.getenv("HONEYPOT_API_KEY")



GUVI_CALLBACK_URL = "https://hackathon.guvi.in/api/updateHoneyPotFinalResult"
MIN_MESSAGES_FOR_CALLBACK = 5

# ============================
# LOAD PHISHING MODEL
# ============================

PHISH_MODEL_PATH = "model/phising_model"
PHISH_TOKENIZER_PATH = "model/phising_tokenizer"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

phish_model = BertForSequenceClassification.from_pretrained(PHISH_MODEL_PATH)
phish_tokenizer = BertTokenizer.from_pretrained(PHISH_TOKENIZER_PATH)

phish_model.to(device)
phish_model.eval()

print("Phishing model loaded")

# ============================
# LOAD AGENT LLM
# ============================

agent_tokenizer = AutoTokenizer.from_pretrained("distilgpt2")
agent_model = AutoModelForCausalLM.from_pretrained("distilgpt2")
agent_model.to(device)
agent_model.eval()

print("Agent LLM loaded")

# ============================
# FLASK APP
# ============================

app = Flask(__name__)

# ============================
# MEMORY STORES
# ============================

conversation_store = {}
intelligence_store = {}
callback_done = {}

# ============================
# VERIFY API KEY
# ============================

def verify_api_key(req):
    key = req.headers.get("x-api-key")
    return key == API_KEY


# ============================
# SCAM DETECTION
# ============================

def detect_scam(text):

    inputs = phish_tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=512
    )

    inputs = {k:v.to(device) for k,v in inputs.items()}

    with torch.no_grad():
        out = phish_model(**inputs)

    probs = torch.softmax(out.logits, dim=1)[0]
    pred = torch.argmax(probs).item()
    conf = probs[pred].item()

    return pred==1, float(conf)

# ============================
# AGENT RESPONSE
# ============================

def generate_agent_reply(history):

    persona = (
        "You are a worried bank customer. "
        "Ask simple short questions. "
        "Do not mention scam or security.\n\n"
    )

    convo=""
    for h in history[-6:]:
        convo+=f"{h['sender']}: {h['text']}\n"

    prompt = persona + convo + "user:"

    inputs = agent_tokenizer(prompt, return_tensors="pt").to(device)

    with torch.no_grad():
        out = agent_model.generate(
            **inputs,
            max_new_tokens=40,
            temperature=0.8,
            do_sample=True,
            pad_token_id=agent_tokenizer.eos_token_id
        )

    txt = agent_tokenizer.decode(out[0], skip_special_tokens=True)
    return txt.split("user:")[-1].strip()

# ============================
# INTELLIGENCE EXTRACTION
# ============================

def extract_intelligence(text):

    return {
        "bankAccounts": re.findall(r"\b\d{9,18}\b", text),
        "upiIds": re.findall(r"[a-zA-Z0-9.\-_]{2,}@[a-zA-Z]{2,}", text),
        "phishingLinks": re.findall(r"https?://\S+", text),
        "phoneNumbers": re.findall(r"\+?\d{10,13}", text),
        "suspiciousKeywords": [
            w for w in ["urgent","verify","blocked","otp","immediately"]
            if w in text.lower()
        ]
    }

# ============================
# SEND CALLBACK
# ============================

def send_callback(session_id):

    payload = {
        "sessionId": session_id,
        "scamDetected": True,
        "totalMessagesExchanged": len(conversation_store[session_id]),
        "extractedIntelligence": intelligence_store[session_id],
        "agentNotes": "Scammer used urgency and payment redirection"
    }

    try:
        requests.post(GUVI_CALLBACK_URL, json=payload, timeout=5)
        callback_done[session_id] = True
        print("Callback sent for", session_id)
    except Exception as e:
        print("Callback failed:", e)

# ============================
# HONEYPOT ENDPOINT
# ============================

@app.route("/honeypot/message", methods=["POST"])
def honeypot_message():

    if not verify_api_key(request):
        return jsonify({"error":"Unauthorized"}), 401

    data = request.get_json()


    if not data or "message" not in data:
        return jsonify({"error":"Invalid request"}),400

    session_id = data.get("sessionId","default")
    text = data["message"].get("text","")

    if session_id not in conversation_store:
        conversation_store[session_id]=[]
        intelligence_store[session_id]={
            "bankAccounts":[],
            "upiIds":[],
            "phishingLinks":[],
            "phoneNumbers":[],
            "suspiciousKeywords":[]
        }
        callback_done[session_id]=False

    conversation_store[session_id].append({
        "sender":"scammer",
        "text":text
    })

    # Extract intelligence
    intel = extract_intelligence(text)
    for k in intel:
        intelligence_store[session_id][k].extend(intel[k])

    scam, conf = detect_scam(text)

    if scam:
        reply = generate_agent_reply(conversation_store[session_id])
    else:
        reply = "Okay."

    conversation_store[session_id].append({
        "sender":"agent",
        "text":reply
    })

    # AUTO CALLBACK
    if scam and not callback_done[session_id]:
        if len(conversation_store[session_id]) >= MIN_MESSAGES_FOR_CALLBACK:
            send_callback(session_id)

    return jsonify({
        "status":"success",
        "scamDetected":scam,
        "confidence":round(conf,3),
        "reply":reply
    })

# ============================
# RUN
# ============================

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    app.run(host="0.0.0.0", port=port)

