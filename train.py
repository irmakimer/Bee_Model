import os
import librosa
import numpy as np
import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score

# === 1. Veri yolları ===
queenbee_present_folder = r'C:\Users\Lenovo\OneDrive\Masaüstü\Mixed\Var'
queenbee_absent_folder = r'C:\Users\Lenovo\OneDrive\Masaüstü\Mixed\Yok'

veri = []

# === 2. MFCC çıkarma ===
def take_mfcc_coefficients(dosya_yolu):
    y, sr = librosa.load(dosya_yolu, sr=None)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    mfcc_mean = np.mean(mfcc, axis=1)
    return mfcc_mean

# === 3. Etiketli verileri oluştur ===
for fname in os.listdir(queenbee_present_folder):
    if fname.endswith(".wav"):
        yol = os.path.join(queenbee_present_folder, fname)
        ozellikler = take_mfcc_coefficients(yol)
        satir = [fname] + list(ozellikler) + [1]
        veri.append(satir)

for fname in os.listdir(queenbee_absent_folder):
    if fname.endswith(".wav"):
        yol = os.path.join(queenbee_absent_folder, fname)
        ozellikler = take_mfcc_coefficients(yol)
        satir = [fname] + list(ozellikler) + [0]
        veri.append(satir)


# === 4. DataFrame ve NaN kontrolü ===
sutunlar = ["dosya_adi"] + [f"mfcc_{i+1}" for i in range(13)] + ["etiket"]
df = pd.DataFrame(veri, columns=sutunlar)

print("NaN içeren satır var mı:", df.isnull().any().any())
print(df[df.isnull().any(axis=1)])

df.to_csv("mfcc_etiketli_veri.csv", index=False)

# === 5. Model Eğitimi ===
X = df.iloc[:, 1:-1]  # dosya_adi hariç, etiket hariç
y = df["etiket"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# === 6. Modeli Kaydet ===
joblib.dump(model, "random_forest_queenbee_model_with_new_data.pkl")

# === 7. Test ve Performans ===
y_pred = model.predict(X_test)
print("\nDoğruluk:", accuracy_score(y_test, y_pred))
print("\nSınıflandırma Raporu:\n", classification_report(y_test, y_pred))
