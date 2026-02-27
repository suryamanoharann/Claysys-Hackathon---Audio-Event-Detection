import os
import torch
import librosa
import numpy as np
import torchvision.models as models
import torch.nn as nn
from flask import Flask, request, jsonify, render_template

app = Flask(__name__)

# --- 1. MODEL SETUP (Now EfficientNet-B0) ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Recreate the EfficientNet-B0 architecture to match your Kaggle training script
model = models.efficientnet_b0(weights=None)

# Adjust input layer for 1-channel Grayscale Spectrograms
model.features[0][0] = nn.Conv2d(1, 32, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)

# Adjust head for 10 UrbanSound classes
in_features = model.classifier[1].in_features
model.classifier = nn.Sequential(
    nn.Dropout(p=0.4),
    nn.Linear(in_features, 10)
)

# Load the weights into the EfficientNet model
# Make sure the filename matches what you downloaded from Kaggle
checkpoint = torch.load("hackaudio2_best.pth", map_location=device) 
model.load_state_dict(checkpoint['model_state_dict'])

model.to(device)
model.eval()

# --- 2. ACTIONABLE ALERTS DICTIONARY ---
CLASS_MAP = {
    0: {"name": "Air Conditioner", "type": "info", "msg": "HVAC system running. Standard operational noise detected."},
    1: {"name": "Car Horn", "type": "warning", "msg": "Vehicle horn detected. Potential traffic congestion or hazard nearby."},
    2: {"name": "Children Playing", "type": "info", "msg": "Children playing. Please proceed with caution in this residential/park zone."},
    3: {"name": "Dog Bark", "type": "info", "msg": "Animal vocalization detected. Routine neighborhood activity."},
    4: {"name": "Drilling", "type": "warning", "msg": "Construction drilling detected. Monitor noise pollution levels in this sector."},
    5: {"name": "Engine Idling", "type": "warning", "msg": "Continuous engine idling detected. Potential emissions protocol violation."},
    6: {"name": "Gun Shot", "type": "danger", "msg": "CRITICAL: Gunfire detected! Immediate security protocols initiated."},
    7: {"name": "Jackhammer", "type": "warning", "msg": "Heavy demolition work in progress. High decibel warning active."},
    8: {"name": "Siren", "type": "danger", "msg": "Emergency vehicle approaching. Please ensure paths are clear."},
    9: {"name": "Street Music", "type": "success", "msg": "Street music detected. Active public space/pedestrian zone."}
}

# --- 3. PREPROCESSING FUNCTION (Now Mel Spectrogram) ---
def process_audio(file_path, sr=22050, duration=4):
    max_samples = sr * duration
    
    # Load audio
    y, sr = librosa.load(file_path, sr=sr)
    
    # Trim silence and pad (Exact match to Kaggle Cell 2)
    y_trimmed, _ = librosa.effects.trim(y, top_db=20)
    if len(y_trimmed) < max_samples:
        y_fixed = np.pad(y_trimmed, (0, max_samples - len(y_trimmed)), mode='constant')
    else:
        y_fixed = y_trimmed[:max_samples]
        
    # Generate Mel Spectrogram (128 bands)
    mel = librosa.feature.melspectrogram(
        y=y_fixed, sr=sr, n_mels=128, fmax=8000, n_fft=2048, hop_length=512
    )
    mel_db = librosa.power_to_db(mel, ref=np.max)
    
    # Normalize between 0 and 1
    mel_db = (mel_db - mel_db.min()) / (mel_db.max() - mel_db.min() + 1e-6)
    
    # Shape for EfficientNet: (Batch=1, Channel=1, Mels=128, Time)
    mel_tensor = torch.tensor(mel_db, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
    return mel_tensor.to(device)

# --- 4. ROUTES ---
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({"error": "No file uploaded"}), 400
        
    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No file selected"}), 400

    temp_path = "temp_audio.wav"
    file.save(temp_path)

    try:
        # Process and Predict
        tensor = process_audio(temp_path)
        with torch.no_grad():
            outputs = model(tensor)
            _, predicted = torch.max(outputs, 1)
            class_idx = predicted.item()

        result_data = CLASS_MAP[class_idx]
        os.remove(temp_path)
        return jsonify(result_data)
        
    except Exception as e:
        if os.path.exists(temp_path):
            os.remove(temp_path)
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, port=5000)