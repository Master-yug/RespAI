import librosa
import numpy as np
import soundfile as sf

def extract_features(file_path):
    """
    Converts cough audio → feature vector
    Works WITHOUT model.
    Later you will match this with your ML pipeline.
    """

    # Load audio
    y, sr = sf.read(file_path)

    # convert to float32
    y = y.astype("float32")

    # Resample to 16k
    if sr != 16000:
        y = librosa.resample(y, orig_sr=sr, target_sr=16000)
        sr = 16000

    # Extract MFCC (40 coefficients)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)

    # Take mean across time to get a 40-dim vector
    features = np.mean(mfcc, axis=1)

    return features.reshape(1, -1)
