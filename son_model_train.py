# train_models.py
import os
import librosa
import numpy as np
import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, f1_score, precision_score, recall_score

def extract_features(file_path):
    # === 1. Ses dosyasını yükle
    y, sr = librosa.load(file_path, sr=None)

    if y is None or len(y) == 0:
        raise ValueError(f"Ses verisi boş veya okunamadı: {file_path}")

    # === 2. MFCC, Delta, Delta-Delta
    n_mfcc = 20
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
    mfcc_delta = librosa.feature.delta(mfcc)
    mfcc_delta2 = librosa.feature.delta(mfcc, order=2)

    # Zaman boyunca ortalamasını al (boyutu sabitlemek için)
    mfcc_mean = np.mean(mfcc, axis=1)
    delta_mean = np.mean(mfcc_delta, axis=1)
    delta2_mean = np.mean(mfcc_delta2, axis=1)

    # === 3. Ekstra ses özellikleri
    zcr = np.mean(librosa.feature.zero_crossing_rate(y))                         # 1
    rms = np.mean(librosa.feature.rms(y=y))                                      # 1
    centroid = np.mean(librosa.feature.spectral_centroid(y=y, sr=sr))           # 1
    bandwidth = np.mean(librosa.feature.spectral_bandwidth(y=y, sr=sr))         # 1
    contrast = np.mean(librosa.feature.spectral_contrast(y=y, sr=sr), axis=1)   # 7
    chroma = np.mean(librosa.feature.chroma_stft(y=y, sr=sr), axis=1)           # 12

    # === 4. Özellikleri birleştir
    features = np.concatenate([
        mfcc_mean,       # 20
        delta_mean,      # 20
        delta2_mean,     # 20
        [zcr, rms, centroid, bandwidth],  # 4
        contrast,        # 7
        chroma           # 12
    ])

    return features

def load_dataset(folder_path, label):
    data = []
    for fname in os.listdir(folder_path):
        if fname.endswith('.wav'):
            path = os.path.join(folder_path, fname)
            try:
                features = extract_features(path)
                data.append(list(features) + [label])
            except Exception as e:
                print(f"Hata: {fname} işlenemedi. {e}")
    return data


def train_model(df, model_path):
    # Veriyi ayır (eğitim ve test seti)
    X = df.iloc[:, :-1]
    y = df['label']
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Modeli oluştur ve eğit
    model = RandomForestClassifier(
        n_estimators=100,
        random_state=50,
        max_depth=200,
        min_samples_split=2,
        min_samples_leaf=1,
        max_features=2,
        bootstrap=True
    )
    model.fit(X_train, y_train)

    # Eğitilen modeli diske kaydet
    joblib.dump(model, model_path)

    # === Test seti üzerindeki başarıyı ölç ===
    y_pred = model.predict(X_test)

    print(f"\n📦 Model adı: {model_path}")
    print("🎯 Test Seti Performansı:")
    print("- Doğruluk (Accuracy):", round(accuracy_score(y_test, y_pred), 4))
    
    # Etiket türüne göre uygun ortalama türünü belirle
    average_type = 'binary' if sorted(y.unique()) == [0, 1] else 'macro'

    print("- F1 Skoru:", round(f1_score(y_test, y_pred, average=average_type), 4))
    print("- Precision:", round(precision_score(y_test, y_pred, average=average_type), 4))
    print("- Recall:", round(recall_score(y_test, y_pred, average=average_type), 4))

    print("\n📊 Sınıf Bazlı Detaylı Rapor:\n")
    target_names = ['Yok', 'Var'] if sorted(y.unique()) == [0, 1] else ['No_Bee', 'Bee']
    print(classification_report(y_test, y_pred, target_names=target_names))

    return model


# === Klasörler ===
bee_path = r'C:\Users\Lenovo\OneDrive\Masaüstü\Bee_Or_No\Bee'
no_bee_path = r'C:\Users\Lenovo\OneDrive\Masaüstü\Bee_Or_No\No_Bee'
queen_path = r'C:\Users\Lenovo\OneDrive\Masaüstü\Testing\Var'
no_queen_path = r'C:\Users\Lenovo\OneDrive\Masaüstü\Testing\Yok'

# Özellik adları (dinamik üret)
feature_names = (
    [f'mfcc_{i+1}' for i in range(20)] +
    [f'delta_{i+1}' for i in range(20)] +
    [f'delta2_{i+1}' for i in range(20)] +
    ['zcr', 'rms', 'centroid', 'bandwidth'] +
    [f'contrast_{i+1}' for i in range(7)] +
    [f'chroma_{i+1}' for i in range(12)]
)

# === Eğitim Verisi Hazırla ===
bee_data = load_dataset(bee_path, 'B')
no_bee_data = load_dataset(no_bee_path, 'N')
df_bee = pd.DataFrame(bee_data + no_bee_data, columns=feature_names + ['label'])

queen_data = load_dataset(queen_path, 1)
no_queen_data = load_dataset(no_queen_path, 0)
df_queen = pd.DataFrame(queen_data + no_queen_data, columns=feature_names + ['label'])

# === Modelleri Eğit ve Kaydet ===
train_model(df_bee, "bee_detector.pkl")
train_model(df_queen, "queen_detector.pkl")