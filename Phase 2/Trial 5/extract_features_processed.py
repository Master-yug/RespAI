"""
Feature extraction from PREPROCESSED cough audio files
and merge with consolidated labels into a single CSV.
"""

import os
import librosa
import numpy as np
import pandas as pd

# -------------------------------------------------------------------
# PATHS (update only if you moved things)
# -------------------------------------------------------------------
AUDIO_DIR = r"C:\Users\Tejash\OneDrive\Desktop\FFmpeg-Builds-latest\coughvid_20211012\Processed v2"
CONSOLIDATED_CSV = r"C:\Users\Tejash\Experiential Learning\Phase 2\Data processing and cleaning Trial 1\Trial 5\OG_labels_filtered.csv"
OUT_CSV = r"C:\Users\Tejash\Experiential Learning\Phase 2\Data processing and cleaning Trial 1\Trial 5\Processed_audio_features_merged.csv"

# -------------------------------------------------------------------
# OPTIONAL: formants (if you had this earlier)
# -------------------------------------------------------------------
try:
    import parselmouth
except ImportError:
    parselmouth = None

# -------------------------------------------------------------------
# FEATURE EXTRACTION FUNCTION  (based on your previous code)
# -------------------------------------------------------------------
def extract_features(audio_path):
    y, sr = librosa.load(audio_path, sr=None)
    result = {}

    # Duration
    result['duration'] = librosa.get_duration(y=y, sr=sr)

    # Silence ratio
    rms = librosa.feature.rms(y=y)[0]
    result['silence_ratio'] = np.sum(rms < 0.01) / len(rms)

    # MFCCs
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)
    for i in range(40):
        result[f'mfcc_{i+1}_mean'] = float(np.mean(mfcc[i]))
        result[f'mfcc_{i+1}_std'] = float(np.std(mfcc[i]))

    # Deltas
    mfcc_delta = librosa.feature.delta(mfcc)
    for i in range(40):
        result[f'delta_mfcc_{i+1}_mean'] = float(np.mean(mfcc_delta[i]))
        result[f'delta_mfcc_{i+1}_std'] = float(np.std(mfcc_delta[i]))

    # Delta-delta
    mfcc_delta2 = librosa.feature.delta(mfcc, order=2)
    for i in range(40):
        result[f'delta2_mfcc_{i+1}_mean'] = float(np.mean(mfcc_delta2[i]))
        result[f'delta2_mfcc_{i+1}_std'] = float(np.std(mfcc_delta2[i]))

    # Chroma
    chroma = librosa.feature.chroma_stft(y=y, sr=sr)
    for i in range(chroma.shape[0]):
        result[f'chroma_{i+1}_mean'] = float(np.mean(chroma[i]))
        result[f'chroma_{i+1}_std'] = float(np.std(chroma[i]))

    # Spectral features
    result['spectral_centroid_mean'] = float(np.mean(librosa.feature.spectral_centroid(y=y, sr=sr)))
    result['spectral_bandwidth_mean'] = float(np.mean(librosa.feature.spectral_bandwidth(y=y, sr=sr)))
    result['spectral_flatness_mean'] = float(np.mean(librosa.feature.spectral_flatness(y=y)))
    result['spectral_rolloff_mean'] = float(np.mean(librosa.feature.spectral_rolloff(y=y, sr=sr)))
    result['spectral_contrast_mean'] = float(np.mean(librosa.feature.spectral_contrast(y=y, sr=sr)))
    result['spectral_flux_mean'] = float(np.mean(librosa.onset.onset_strength(y=y, sr=sr)))

    # RMS
    result['rms_mean'] = float(np.mean(rms))
    result['rms_std'] = float(np.std(rms))

    # Zero-Crossing Rate
    zcr = librosa.feature.zero_crossing_rate(y)
    result['zcr_mean'] = float(np.mean(zcr))
    result['zcr_std'] = float(np.std(zcr))

    # Tonnetz
    try:
        tonnetz = librosa.feature.tonnetz(y=librosa.effects.harmonic(y), sr=sr)
        for i in range(tonnetz.shape[0]):
            result[f'tonnetz_{i+1}_mean'] = float(np.mean(tonnetz[i]))
            result[f'tonnetz_{i+1}_std'] = float(np.std(tonnetz[i]))
    except Exception:
        for i in range(6):
            result[f'tonnetz_{i+1}_mean'] = np.nan
            result[f'tonnetz_{i+1}_std'] = np.nan

    # Pitch Tracking (F0)
    pitches, mag = librosa.piptrack(y=y, sr=sr)
    mask = mag > np.median(mag)
    pitches = pitches[mask]
    if len(pitches) > 0:
        result['pitch_mean'] = float(np.mean(pitches))
        result['pitch_std'] = float(np.std(pitches))
        result['pitch_max'] = float(np.max(pitches))
        result['pitch_min'] = float(np.min(pitches))
    else:
        result['pitch_mean'] = 0.0
        result['pitch_std'] = 0.0
        result['pitch_max'] = 0.0
        result['pitch_min'] = 0.0

    # Mel Spectrogram band means
    mel = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=40)
    for i in range(mel.shape[0]):
        result[f'mel_{i+1}_mean'] = float(np.mean(mel[i]))
        result[f'mel_{i+1}_std'] = float(np.std(mel[i]))

    # Formants (optional)
    if parselmouth:
        try:
            snd = parselmouth.Sound(audio_path)
            formant = snd.to_formant_burg()
            for nform in range(1, 5):
                formant_vals = []
                for t in np.linspace(0, snd.duration, num=100):
                    try:
                        val = formant.get_value_at_time(nform, t)
                        if val:
                            formant_vals.append(val)
                    except Exception:
                        pass
                result[f'formant{nform}_mean'] = float(np.mean(formant_vals)) if formant_vals else np.nan
                result[f'formant{nform}_std'] = float(np.std(formant_vals)) if formant_vals else np.nan
        except Exception:
            for nform in range(1, 5):
                result[f'formant{nform}_mean'] = np.nan
                result[f'formant{nform}_std'] = np.nan

    return result

# -------------------------------------------------------------------
# MAIN: EXTRACT FEATURES AND MERGE WITH METADATA
# -------------------------------------------------------------------
def main():
    print("="*80)
    print("FEATURE EXTRACTION FROM PREPROCESSED AUDIO + MERGE WITH LABELS")
    print("="*80)

    # Load consolidated CSV
    meta = pd.read_csv(CONSOLIDATED_CSV)
    print(f"Loaded consolidated CSV with {len(meta)} rows")

    # Column with filenames
    filename_col = "file_features"
    if filename_col not in meta.columns:
        raise ValueError(f"Column '{filename_col}' not found in consolidated CSV")

    # Map for faster join later
    meta[filename_col] = meta[filename_col].astype(str).str.strip()

    # List all processed audio files
    audio_files = [f for f in os.listdir(AUDIO_DIR) if f.lower().endswith(".wav")]
    audio_set = set(audio_files)
    print(f"Found {len(audio_files)} processed audio files in folder")

    features_list = []

    for idx, fname in enumerate(audio_files, 1):
        audio_path = os.path.join(AUDIO_DIR, fname)
        print(f"{idx}/{len(audio_files)} Extracting: {fname}")

        try:
            feats = extract_features(audio_path)
            feats['file_features'] = fname
            features_list.append(feats)
        except Exception as e:
            print(f"  Failed on {fname}: {e}")
            continue

    feat_df = pd.DataFrame(features_list)
    print(f"\nExtracted features for {len(feat_df)} files")

    # Merge with metadata on file_features
    merged = pd.merge(meta, feat_df, on="file_features", how="inner")
    print(f"Merged rows (features + labels): {len(merged)}")

    # Save to CSV
    merged.to_csv(OUT_CSV, index=False)
    print(f"\nSaved merged features+labels to:\n{OUT_CSV}")

if __name__ == "__main__":
    main()
