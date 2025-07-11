from flask import Flask, request, jsonify
import joblib
import numpy as np
import librosa
import tempfile
import os
from flask_cors import CORS

app = Flask(__name__)
CORS(app)  # Tüm domainlerden istek kabul et


def extract_features(file_path):
    y, sr = librosa.load(file_path, sr=None)
    if y is None or len(y) == 0:
        raise ValueError("Ses verisi boş veya okunamadı.")

    n_mfcc = 20
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
    mfcc_delta = librosa.feature.delta(mfcc)
    mfcc_delta2 = librosa.feature.delta(mfcc, order=2)

    mfcc_mean = np.mean(mfcc, axis=1)
    delta_mean = np.mean(mfcc_delta, axis=1)
    delta2_mean = np.mean(mfcc_delta2, axis=1)

    zcr = np.mean(librosa.feature.zero_crossing_rate(y))
    rms = np.mean(librosa.feature.rms(y=y))
    centroid = np.mean(librosa.feature.spectral_centroid(y=y, sr=sr))
    bandwidth = np.mean(librosa.feature.spectral_bandwidth(y=y, sr=sr))
    contrast = np.mean(librosa.feature.spectral_contrast(y=y, sr=sr), axis=1)
    chroma = np.mean(librosa.feature.chroma_stft(y=y, sr=sr), axis=1)

    features = np.concatenate([
        mfcc_mean, delta_mean, delta2_mean,
        [zcr, rms, centroid, bandwidth],
        contrast, chroma
    ])

    return features

# Modelleri yükle
bee_model = joblib.load("bee_detector.pkl")
queen_model = joblib.load("queen_detector.pkl")


@app.route('/')
def home():
    return "Model Test API çalışıyor!"


@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'Dosya bulunamadı! "file" anahtarı ile gönderin.'}), 400

    file = request.files['file']

    if file.filename == '':
        return jsonify({'error': 'Dosya adı boş.'}), 400

    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
            file.save(tmp.name)
            tmp_path = tmp.name

        features = extract_features(tmp_path).reshape(1, -1)

        # Arı tahmini
        bee_pred = bee_model.predict(features)[0]
        bee_prob = bee_model.predict_proba(features).max()

        result = {
            'bee_prediction': bee_pred,
            'bee_confidence': float(bee_prob),
            'bee_status': "Arı sesi var" if bee_pred == 'B' else "Arı sesi yok"
        }

        # Kraliçe tahmini (sadece arı varsa)
        if bee_pred == 'B':
            queen_pred = queen_model.predict(features)[0]
            queen_prob = queen_model.predict_proba(features).max()
            result.update({
                'queen_prediction': int(queen_pred),
                'queen_confidence': float(queen_prob),
                'queen_status': "Kraliçe arı sesi var" if queen_pred == 1 else "Kraliçe arı sesi yok"
            })
        else:
            result.update({
                'queen_prediction': None,
                'queen_confidence': None,
                'queen_status': "Arı sesi yok, kraliçe arı analizi yapılmadı"
            })

        os.remove(tmp_path)
        return jsonify(result)

    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True, use_reloader=False)

