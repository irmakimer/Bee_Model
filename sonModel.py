import os
import librosa
import numpy as np
import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

# === 1. Gelişmiş Özellik Çıkarma Fonksiyonu ===
def extract_features(file_path):
    y, sr = librosa.load(file_path, sr=None)

    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    mfcc_mean = np.mean(mfcc, axis=1)

    zcr = np.mean(librosa.feature.zero_crossing_rate(y))
    rms = np.mean(librosa.feature.rms(y=y))
    centroid = np.mean(librosa.feature.spectral_centroid(y=y, sr=sr))
    bandwidth = np.mean(librosa.feature.spectral_bandwidth(y=y, sr=sr))
    contrast = np.mean(librosa.feature.spectral_contrast(y=y, sr=sr), axis=1)
    chroma = np.mean(librosa.feature.chroma_stft(y=y, sr=sr), axis=1)

    features = np.concatenate([
        mfcc_mean,
        [zcr, rms, centroid, bandwidth],
        contrast,
        chroma
    ])
    return features

# === 2. Eğitim veri setleri için yollar ===
bee_path = r'C:\Users\Lenovo\OneDrive\Masaüstü\Bee_Or_No\Bee'
no_bee_path = r'C:\Users\Lenovo\OneDrive\Masaüstü\Bee_Or_No\No_Bee'
queen_path = r'C:\Users\Lenovo\OneDrive\Masaüstü\Mixed\Var'
no_queen_path = r'C:\Users\Lenovo\OneDrive\Masaüstü\Mixed\Yok'

# === 3. Eğitim verilerini oku ===
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

# Tüm özellik isimlerini oluştur
feature_names = (
    [f'mfcc_{i+1}' for i in range(13)] +
    ['zcr', 'rms', 'centroid', 'bandwidth'] +
    [f'contrast_{i+1}' for i in range(7)] +
    [f'chroma_{i+1}' for i in range(12)]
)

# Arı sesi modeli için veri
bee_data = load_dataset(bee_path, 'B')
no_bee_data = load_dataset(no_bee_path, 'N')
df_bee = pd.DataFrame(bee_data + no_bee_data, columns=feature_names + ['label'])

# Kraliçe arı modeli için veri
queen_data = load_dataset(queen_path, 1)
no_queen_data = load_dataset(no_queen_path, 0)
df_queen = pd.DataFrame(queen_data + no_queen_data, columns=feature_names + ['label'])

# === 4. Model eğit ===
def train_model(df, model_path):
    X = df.iloc[:, :-1]
    y = df['label']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    joblib.dump(model, model_path)
    y_pred = model.predict(X_test)
    print(f"\nModel: {model_path}")
    print(classification_report(y_test, y_pred))
    return model

bee_model = train_model(df_bee, "bee_detector.pkl")
queen_model = train_model(df_queen, "queen_detector.pkl")

# === 5. Test için ses dosyasını analiz et ===
def classify_audio(audio_path):
    try:
        features_array = extract_features(audio_path).reshape(1, -1)
        features = pd.DataFrame(features_array, columns=feature_names)

        bee_model = joblib.load("bee_detector.pkl")
        queen_model = joblib.load("queen_detector.pkl")

        bee_result = bee_model.predict(features)[0]
        if bee_result == 'N':
            return "Arı yok"
        else:
            queen_result = queen_model.predict(features)[0]
            if queen_result == 1:
                return "Arı var, kraliçe arı var"
            else:
                return "Arı var, kraliçe arı yok"
    except Exception as e:
        return f"Hata oluştu: {e}"



# === 6. Test klasöründeki tüm dosyaları test et ===
test_klasoru = r"C:\Users\Lenovo\OneDrive\Masaüstü\Test"

for dosya in os.listdir(test_klasoru):
    if dosya.endswith(".wav"):
        yol = os.path.join(test_klasoru, dosya)
        sonuc = classify_audio(yol)
        print(f"\n>> {dosya} Tahmin:", sonuc)
