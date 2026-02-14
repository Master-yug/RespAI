# RespAI
RespAI is a cough-audio screening demo that pairs a browser-based recorder with a Flask API to estimate respiratory risk using classic audio features and a trained ML model. The current build focuses on end-to-end capture, upload, and prediction flow.

## Status
- Frontend: single-page HTML experience with consent, recording, and results
- Backend: Flask API for feature extraction and model inference
- Node/Express scaffold: present but not yet implemented (files are empty)

## Features
- In-browser cough recording via MediaRecorder
- Upload to backend for feature extraction and inference
- Risk display with confidence and diagnosis label
- Simple safety notice and consent flow

## Getting Started

### Prerequisites
- Python 3.9+ recommended
- A modern browser with microphone access

### Backend (Flask)
1. Create a virtual environment and install deps:
	 ```bash
	 python -m venv .venv
	 .venv\Scripts\activate
	 pip install -r backend/requirements.txt
	 ```
2. Start the API server:
	 ```bash
	 python backend/main.py
	 ```
	 The server runs at `http://localhost:5000`.

### Frontend (Static HTML)
Open [frontend/resp.html](frontend/resp.html) in a local web server (recommended) so the browser allows microphone access. If you use VS Code, the Live Server extension works well.

The page sends audio to `http://localhost:5000/predict`, so the Flask server must be running.

## API

### POST /predict
Accepts a multipart form upload under `file` and returns diagnosis + confidence.

Example:
```bash
curl -X POST http://localhost:5000/predict \
	-F "file=@sample.wav"
```

Response shape:
```json
{
	"diagnosis": "Pneumonia",
	"confidence": 0.91,
	"probabilities": {
		"Asthma": 0.02,
		"Pneumonia": 0.91,
		"Healthy": 0.07
	}
}
```

## Model Notes
- `backend/models/` includes `diagnosis_model.pkl`, `scaler.pkl`, `label_encoder.pkl`, and `feature_columns.json`.
- Feature extraction uses `backend/utils/audio_features.py` (MFCCs, deltas, chroma, spectral, mel, formants).
- If model artifacts are missing, the backend will run in simulation mode and return errors on `/predict`.

## Disclaimers
RespAI is a demo and not a clinical diagnostic tool. Always consult a licensed medical professional for medical advice.

## Roadmap Ideas
- Wire up the Node/Express scaffold or remove it
- Add model versioning and provenance
- Improve error handling for unsupported audio formats
- Add unit tests for feature extraction and API validation
