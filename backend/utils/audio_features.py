import librosa
import numpy as np

try:
    import parselmouth
except ImportError:
    parselmouth = None


def extract_features(audio_path):
    """
    Extract audio features from a cough audio file.
    Returns a dictionary of feature names and values.
    """
    # Load audio (librosa can handle wav/webm etc. via audioread)
    y, sr = librosa.load(audio_path, sr=None)
    result = {}

    # ---------------- Basic energy / duration ----------------
    result["duration"] = librosa.get_duration(y=y, sr=sr)

    rms = librosa.feature.rms(y=y)[0]
    result["silence_ratio"] = float(np.sum(rms < 0.01) / len(rms))

    # ---------------- MFCCs + deltas ----------------
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)
    for i in range(40):
        result[f"mfcc_{i+1}_mean"] = float(np.mean(mfcc[i]))
        result[f"mfcc_{i+1}_std"] = float(np.std(mfcc[i]))

    mfcc_delta = librosa.feature.delta(mfcc)
    for i in range(40):
        result[f"delta_mfcc_{i+1}_mean"] = float(np.mean(mfcc_delta[i]))
        result[f"delta_mfcc_{i+1}_std"] = float(np.std(mfcc_delta[i]))

    mfcc_delta2 = librosa.feature.delta(mfcc, order=2)
    for i in range(40):
        result[f"delta2_mfcc_{i+1}_mean"] = float(np.mean(mfcc_delta2[i]))
        result[f"delta2_mfcc_{i+1}_std"] = float(np.std(mfcc_delta2[i]))

    # ---------------- Chroma ----------------
    chroma = librosa.feature.chroma_stft(y=y, sr=sr)
    for i in range(chroma.shape[0]):
        result[f"chroma_{i+1}_mean"] = float(np.mean(chroma[i]))
        result[f"chroma_{i+1}_std"] = float(np.std(chroma[i]))

    # ---------------- Spectral features ----------------
    result["spectral_centroid_mean"] = float(
        np.mean(librosa.feature.spectral_centroid(y=y, sr=sr))
    )
    result["spectral_bandwidth_mean"] = float(
        np.mean(librosa.feature.spectral_bandwidth(y=y, sr=sr))
    )
    result["spectral_flatness_mean"] = float(
        np.mean(librosa.feature.spectral_flatness(y=y))
    )
    result["spectral_rolloff_mean"] = float(
        np.mean(librosa.feature.spectral_rolloff(y=y, sr=sr))
    )
    result["spectral_contrast_mean"] = float(
        np.mean(librosa.feature.spectral_contrast(y=y, sr=sr))
    )
    result["spectral_flux_mean"] = float(
        np.mean(librosa.onset.onset_strength(y=y, sr=sr))
    )

    # ---------------- RMS + ZCR ----------------
    result["rms_mean"] = float(np.mean(rms))
    result["rms_std"] = float(np.std(rms))

    zcr = librosa.feature.zero_crossing_rate(y)
    result["zcr_mean"] = float(np.mean(zcr))
    result["zcr_std"] = float(np.std(zcr))

    # ---------------- Tonnetz ----------------
    try:
        tonnetz = librosa.feature.tonnetz(y=librosa.effects.harmonic(y), sr=sr)
        for i in range(tonnetz.shape[0]):
            result[f"tonnetz_{i+1}_mean"] = float(np.mean(tonnetz[i]))
            result[f"tonnetz_{i+1}_std"] = float(np.std(tonnetz[i]))
    except Exception:
        # If tonnetz cannot be computed, keep same keys with NaN
        for i in range(6):
            result[f"tonnetz_{i+1}_mean"] = np.nan
            result[f"tonnetz_{i+1}_std"] = np.nan

    # ---------------- Pitch (F0) ----------------
    pitches, mag = librosa.piptrack(y=y, sr=sr)
    mask = mag > np.median(mag)
    pitches = pitches[mask]
    if pitches.size > 0:
        result["pitch_mean"] = float(np.mean(pitches))
        result["pitch_std"] = float(np.std(pitches))
        result["pitch_max"] = float(np.max(pitches))
        result["pitch_min"] = float(np.min(pitches))
    else:
        result["pitch_mean"] = 0.0
        result["pitch_std"] = 0.0
        result["pitch_max"] = 0.0
        result["pitch_min"] = 0.0

    # ---------------- Mel bands ----------------
    mel = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=40)
    for i in range(mel.shape[0]):
        result[f"mel_{i+1}_mean"] = float(np.mean(mel[i]))
        result[f"mel_{i+1}_std"] = float(np.std(mel[i]))

    # ---------------- Formants (optional, robust) ----------------
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
                if formant_vals:
                    result[f"formant{nform}_mean"] = float(np.mean(formant_vals))
                    result[f"formant{nform}_std"] = float(np.std(formant_vals))
                else:
                    result[f"formant{nform}_mean"] = np.nan
                    result[f"formant{nform}_std"] = np.nan
        except Exception:
            # File format not supported by Praat; keep NaNs for consistency
            for nform in range(1, 5):
                result[f"formant{nform}_mean"] = np.nan
                result[f"formant{nform}_std"] = np.nan
    else:
        for nform in range(1, 5):
            result[f"formant{nform}_mean"] = np.nan
            result[f"formant{nform}_std"] = np.nan

    return result
