import os
import librosa
import soundfile as sf

def trim_from_middle(y, sr, duration_sec):
    target_samples = int(duration_sec * sr)
    total_samples = len(y)

    if total_samples <= target_samples:
        return y
    else:
        start = (total_samples - target_samples) // 2
        end = start + target_samples
        return y[start:end]

def ensure_dirs(base_output_dir, durations):
    for dur in durations:
        os.makedirs(os.path.join(base_output_dir, f"{dur}s"), exist_ok=True)

def process_all_wav_files(input_dir, base_output_dir):
    durations = [3, 5, 8]
    ensure_dirs(base_output_dir, durations)

    for filename in os.listdir(input_dir):
        if filename.endswith(".wav"):
            file_path = os.path.join(input_dir, filename)
            y, sr = librosa.load(file_path, sr=None)
            duration_sec = len(y) / sr
            base_name = os.path.splitext(filename)[0]

            if duration_sec < 3:
                print(f"{filename}: <3s, atlandı.")
                continue

            # 3 saniye
            if duration_sec >= 3:
                y_3 = trim_from_middle(y, sr, 3)
                out_path_3 = os.path.join(base_output_dir, "3s", 
                                          f"{base_name}_middle_3s.wav")
                sf.write(out_path_3, y_3, sr)
                print(f"{filename}: 3s middle kırpıldı.")

            # 5 saniye
            if duration_sec >= 5:
                y_5 = trim_from_middle(y, sr, 5)
                out_path_5 = os.path.join(base_output_dir, "5s",
                                          f"{base_name}_middle_5s.wav")
                sf.write(out_path_5, y_5, sr)
                print(f"{filename}: 5s middle kırpıldı.")

            # 8 saniye
            if duration_sec >= 8:
                y_8 = trim_from_middle(y, sr, 8)
                out_path_8 = os.path.join(base_output_dir, "8s", 
                                          f"{base_name}_middle_8s.wav")
                sf.write(out_path_8, y_8, sr)
                print(f"{filename}: 8s middle kırpıldı.")

# === Örnek kullanım ===
input_folder = r'C:\Users\Lenovo\OneDrive\Masaüstü\Bee_Or_No\Bee'
output_base_folder = r'C:\Users\Lenovo\OneDrive\Masaüstü\Bee'
process_all_wav_files(input_folder, output_base_folder)
