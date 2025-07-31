# test.py
import os
import librosa
import numpy as np
import pandas as pd
import joblib
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

def extract_features(file_path):
    y, sr = librosa.load(file_path, sr=None)
    if y is None or len(y) == 0:
        raise ValueError(f"Ses verisi boş veya okunamadı: {file_path}")

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

feature_names = (
    [f'mfcc_{i+1}' for i in range(20)] +
    [f'delta_{i+1}' for i in range(20)] +
    [f'delta2_{i+1}' for i in range(20)] +
    ['zcr', 'rms', 'centroid', 'bandwidth'] +
    [f'contrast_{i+1}' for i in range(7)] +
    [f'chroma_{i+1}' for i in range(12)]
)

bee_model = joblib.load("bee_detector.pkl")
queen_model = joblib.load("queen_detector.pkl")

# Test klasörü
test_klasoru = r"C:\Users\Lenovo\OneDrive\Masaüstü\proje\Bee_Sound\Data\Test_Verileri"

true_bee_labels = []
pred_bee_labels = []

true_queen_labels = []
pred_queen_labels = []

print("=== Tahminler ===")
for dosya in os.listdir(test_klasoru):
    if dosya.endswith(".wav"):
        yol = os.path.join(test_klasoru, dosya)
        try:
            features_array = extract_features(yol).reshape(1, -1)
            features = pd.DataFrame(features_array, columns=feature_names)

            bee_pred = bee_model.predict(features)[0]
            bee_true = 'B' if 'B' in dosya else 'N'
            true_bee_labels.append(bee_true)
            pred_bee_labels.append(bee_pred)

            if bee_pred == 'B':
                queen_pred = queen_model.predict(features)[0]

                # Daha esnek etiket tespiti
                queen_true = 1 if any(keyword in dosya.lower() for keyword in ['var', 'queen', 'kraliçe']) else 0

                true_queen_labels.append(queen_true)
                pred_queen_labels.append(queen_pred)

                print(f"{dosya} -> Arı var, kraliçe {'var' if queen_pred == 1 else 'yok'}")
            else:
                print(f"{dosya} -> Arı yok")

        except Exception as e:
            print(f"{dosya} için hata: {e}")

# === Arı Sınıflandırma Metrikleri ===
print("\n==== Arı Sınıflandırma Metrikleri ====")
print(f"Accuracy: {accuracy_score(true_bee_labels, pred_bee_labels):.2f}")
print(f"Precision: {precision_score(true_bee_labels, pred_bee_labels, pos_label='B', zero_division=0):.2f}")
print(f"Recall: {recall_score(true_bee_labels, pred_bee_labels, pos_label='B', zero_division=0):.2f}")
print(f"F1 Score: {f1_score(true_bee_labels, pred_bee_labels, pos_label='B', zero_division=0):.2f}")

# === Kraliçe Arı Metrikleri (sadece arı tespit edilenler için) ===
if len(true_queen_labels) > 0 and len(set(true_queen_labels)) == 2:
    print("\n==== Kraliçe Arı Sınıflandırma Metrikleri ====")
    print(f"Accuracy: {accuracy_score(true_queen_labels, pred_queen_labels):.2f}")
    print(f"Precision: {precision_score(true_queen_labels, pred_queen_labels, pos_label=1, zero_division=0):.2f}")
    print(f"Recall: {recall_score(true_queen_labels, pred_queen_labels, pos_label=1, zero_division=0):.2f}")
    print(f"F1 Score: {f1_score(true_queen_labels, pred_queen_labels, pos_label=1, zero_division=0):.2f}")
else:
    print("\nKraliçe arı metrikleri hesaplanamadı: Yeterli sınıf çeşitliliği yok.")

# Etiket kontrolü
print("\n[Debug] Kraliçe arı gerçek etiketlerinin dağılımı:")
print(pd.Series(true_queen_labels).value_counts())
