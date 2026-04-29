from fastapi import FastAPI, UploadFile, File
import numpy as np
import joblib
import librosa
import tempfile
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # test ke liye ok
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def home():
    return {"message": "API is working"}

model = joblib.load("emotion_model.pkl")
scaler = joblib.load("scaler.pkl")

def extract_features(audio, sr):
    mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=15)
    mfcc_mean = np.mean(mfcc, axis=1)
    return mfcc_mean.reshape(1, -1)

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    
    content = await file.read()
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as f:
        f.write(content)
        path = f.name

    audio, sr = librosa.load(path, sr=16000)

    features = extract_features(audio, sr)
    features = scaler.transform(features)

    pred = model.predict(features)[0]

    return {
        "prediction": pred
    }
