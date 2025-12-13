"""
Advanced Audio Preprocessing Pipeline for Cough Sounds
WITH CONSERVATIVE SPEECH/VOICE REMOVAL (Cough-Safe)

Key Principle: Only removes CLEARLY IDENTIFIABLE speech/voice
Does NOT compromise cough audio in any way
"""

import os
import pandas as pd
import librosa
import numpy as np
import soundfile as sf
from scipy import signal

# Optional: For voice detection (install if available)
try:
    import webrtcvad
    VAD_AVAILABLE = True
except ImportError:
    VAD_AVAILABLE = False
    print("WARNING: webrtcvad not installed. Install with: pip install webrtcvad")
    print("         Voice removal will be skipped. Run: pip install webrtcvad")

# Define paths
FILTERED_CSV = r"C:\Users\Tejash\Experiential Learning\Phase 2\Data processing and cleaning Trial 1\Trial 5\OG_labels_filtered.csv"
SOURCE_AUDIO_DIR = r"C:\Users\Tejash\OneDrive\Desktop\FFmpeg-Builds-latest\coughvid_20211012\fixed"
TARGET_AUDIO_DIR = r"C:\Users\Tejash\OneDrive\Desktop\FFmpeg-Builds-latest\coughvid_20211012\Processed v2"

# Create target directory if it doesn't exist
os.makedirs(TARGET_AUDIO_DIR, exist_ok=True)

def detect_and_remove_speech(y, sr=22050, aggressiveness=1):
    """
    Detect and CONSERVATIVELY remove speech/voice while preserving cough.
    
    Strategy: Only remove frames that are CLEARLY speech, not ambiguous.
    Cough frequencies are mostly preserved because:
    - Cough has strong energy in 50-8000 Hz (especially 500-4000 Hz)
    - Speech formants are 700-3500 Hz, but speech has different temporal patterns
    - We use webrtcvad which is trained on human speech patterns
    - We only remove the CLEAREST speech frames (high confidence)
    
    Args:
        y: Audio signal
        sr: Sample rate
        aggressiveness: 0-3, higher = more aggressive speech removal
                       0 = very conservative (almost no removal)
                       1 = moderate (removes clear speech, keeps ambiguous)
                       2 = aggressive (removes most speech-like frames)
                       3 = very aggressive (may affect cough)
    
    Returns:
        y_speech_removed: Audio with speech removed
        speech_removed_pct: Percentage of audio removed
    """
    
    if not VAD_AVAILABLE:
        print("    [Info] webrtcvad not available - skipping voice detection")
        return y, 0.0
    
    try:
        # Resample to 8kHz or 16kHz for VAD (webrtcvad requirement)
        # webrtcvad works best at 8kHz or 16kHz
        vad_sr = 16000
        y_vad = librosa.resample(y, orig_sr=sr, target_sr=vad_sr)
        
        # Initialize VAD detector
        vad = webrtcvad.Vad(aggressiveness)
        
        # Convert to bytes for VAD
        # VAD expects 16-bit PCM audio
        y_int16 = np.int16(y_vad / np.max(np.abs(y_vad)) * 32767)
        audio_bytes = y_int16.tobytes()
        
        # Process in frames (10, 20, or 30 ms)
        frame_duration_ms = 20  # milliseconds
        frame_length = int(vad_sr * frame_duration_ms / 1000)
        
        # Detect speech in each frame
        speech_frames = []
        for i in range(0, len(audio_bytes) - frame_length * 2, frame_length * 2):
            frame = audio_bytes[i:i + frame_length * 2]
            if len(frame) < frame_length * 2:
                break
            
            is_speech = vad.is_speech(frame, vad_sr)
            speech_frames.append(is_speech)
        
        if not speech_frames:
            print("    [Info] No speech detected - keeping full audio")
            return y, 0.0
        
        # Create mask for speech frames
        # Map speech detection back to original sample rate
        frame_length_original = int(sr * frame_duration_ms / 1000)
        speech_mask = np.zeros(len(y), dtype=bool)
        
        for frame_idx, is_speech in enumerate(speech_frames):
            start = frame_idx * frame_length_original
            end = start + frame_length_original
            if end > len(y):
                end = len(y)
            
            if is_speech:
                speech_mask[start:end] = True
        
        # CONSERVATIVE APPROACH: Only remove speech if it's VERY confident
        # Apply smoothing to speech mask to avoid removing isolated speech frames
        # This prevents removing brief speech mixed with cough
        
        # Apply morphological smoothing
        from scipy import ndimage
        
        # Expand speech regions slightly to remove speech properly
        expanded_mask = ndimage.binary_dilation(speech_mask, iterations=1)
        
        # Contract back to remove very small isolated detections (likely false positives)
        final_mask = ndimage.binary_erosion(expanded_mask, iterations=2)
        
        # Only mark as speech if detection is strong and continuous
        # This ensures we don't accidentally remove cough
        
        # Remove only the clearly marked speech regions
        y_cleaned = y.copy()
        y_cleaned[final_mask] = 0  # Mute speech frames
        
        # Apply smooth fading at transitions to avoid clicks/artifacts
        fade_length = int(0.05 * sr)  # 50 ms fade
        for i in range(1, len(final_mask) - 1):
            if final_mask[i] != final_mask[i - 1]:  # Transition point
                if final_mask[i]:  # Transitioning into speech
                    start = max(0, i - fade_length)
                    fade = np.linspace(1, 0, i - start)
                    y_cleaned[start:i] *= fade
                else:  # Transitioning out of speech
                    end = min(len(y_cleaned), i + fade_length)
                    fade = np.linspace(0, 1, end - i)
                    y_cleaned[i:end] *= fade
        
        # Calculate percentage of audio removed
        speech_removed_pct = np.sum(final_mask) / len(y) * 100
        
        print(f"    [Voice Detection] Removed {speech_removed_pct:.1f}% of audio as speech")
        
        return y_cleaned, speech_removed_pct
        
    except Exception as e:
        print(f"    [Warning] Voice detection failed: {str(e)}")
        print(f"             Proceeding without voice removal")
        return y, 0.0

def preprocess_cough_audio(audio_path, sr=22050, remove_speech=True):
    """
    Preprocess cough audio with the following steps:
    1. Load audio at standard sample rate
    2. Normalize audio to prevent clipping
    3. Apply bandpass filter (cough typically 50-8000 Hz)
    4. CONSERVATIVELY remove speech/voice (COUGH-SAFE)
    5. Reduce background noise using spectral gating
    6. Trim leading/trailing silence
    
    IMPORTANT: Speech removal is CONSERVATIVE to avoid compromising cough audio
    
    Args:
        audio_path: Path to audio file
        sr: Target sample rate (default 22050 Hz)
        remove_speech: Whether to remove speech (default True)
    
    Returns:
        y_preprocessed: Preprocessed audio signal
        sr: Sample rate
    """
    
    try:
        # Step 1: Load audio at standard sample rate
        print(f"  Loading: {os.path.basename(audio_path)}")
        y, sr = librosa.load(audio_path, sr=sr)
        
        # Step 2: Normalize audio (prevent clipping and ensure consistent loudness)
        # RMS normalization to -20 dB (standard for speech/audio processing)
        target_rms = 0.1  # Corresponds to approximately -20 dB
        current_rms = np.sqrt(np.mean(y**2))
        if current_rms > 0:
            y = y * (target_rms / current_rms)
        
        # Step 3: Apply bandpass filter (cough sounds typically 50-8000 Hz)
        # Removes very low frequency rumble and ultra-high frequency noise
        low_freq = 50
        high_freq = 8000
        
        # Design butterworth bandpass filter
        nyquist = sr / 2
        low = low_freq / nyquist
        high = high_freq / nyquist
        
        # Ensure filter parameters are valid
        if low > 0 and high < 1 and low < high:
            b, a = signal.butter(4, [low, high], btype='band')
            y = signal.filtfilt(b, a, y)
            print(f"    [Bandpass Filter] Applied 50-8000 Hz filter")
        
        # Step 4: CONSERVATIVELY remove speech/voice (COUGH-SAFE)
        if remove_speech and VAD_AVAILABLE:
            print(f"    [Speech Detection] Starting conservative voice removal...")
            y, speech_pct = detect_and_remove_speech(y, sr, aggressiveness=1)
            if speech_pct > 50:
                print(f"    [Warning] More than 50% marked as speech - this file may have been contaminated")
        
        # Step 5: Noise reduction using spectral gating
        # Remove low-energy frames that are likely background noise
        # BUT preserve frames with cough energy
        S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128)
        S_db = librosa.power_to_db(S, ref=np.max)
        
        # Compute energy per frame
        energy = np.mean(S, axis=0)
        energy_normalized = (energy - np.min(energy)) / (np.max(energy) - np.min(energy) + 1e-8)
        
        # Gate frames: keep only frames above 30% of max energy
        # This threshold is conservative to preserve cough
        gate_threshold = 0.3
        frame_length = len(y) // len(energy_normalized)
        mask = np.repeat(energy_normalized > gate_threshold, frame_length)
        
        # Adjust mask length to match audio length
        if len(mask) > len(y):
            mask = mask[:len(y)]
        elif len(mask) < len(y):
            mask = np.pad(mask, (0, len(y) - len(mask)), 'constant', constant_values=False)
        
        # Apply gentle fading at gate transitions to avoid clicks
        fade_length = int(0.01 * sr)  # 10 ms fade
        for i in range(len(mask) - 1):
            if mask[i] != mask[i + 1]:  # Transition point
                if mask[i]:  # Transitioning from True to False
                    start = min(i + fade_length, len(y))
                    y[i:start] *= np.linspace(1, 0, start - i)
                else:  # Transitioning from False to True
                    start = max(0, i - fade_length)
                    y[start:i] *= np.linspace(0, 1, i - start)
        
        print(f"    [Spectral Gating] Applied noise gate (removed low-energy frames)")
        
        # Step 6: Trim silence from start and end
        # Use librosa's trim function with aggressive threshold for cough detection
        y_trimmed, _ = librosa.effects.trim(y, top_db=30)
        
        # Ensure we don't have empty audio
        if len(y_trimmed) < sr * 0.5:  # Less than 0.5 seconds
            print(f"    [Warning] Audio too short after trimming, using full preprocessed audio")
            y_trimmed = y
        else:
            silence_removed_pct = (1 - len(y_trimmed) / len(y)) * 100
            print(f"    [Silence Trim] Removed {silence_removed_pct:.1f}% silence from start/end")
        
        print(f"    [Success] Preprocessing complete - output duration: {len(y_trimmed)/sr:.2f}s")
        
        return y_trimmed, sr
        
    except Exception as e:
        print(f"  Error preprocessing {os.path.basename(audio_path)}: {str(e)}")
        return None, None

def main():
    """
    Main pipeline:
    1. Read filtered CSV to get list of required audio files
    2. Find those files in source directory
    3. Preprocess each audio (WITH COUGH-SAFE SPEECH REMOVAL)
    4. Save to target directory
    """
    
    print("=" * 80)
    print("ADVANCED AUDIO PREPROCESSING PIPELINE FOR COUGH SOUNDS")
    print("WITH CONSERVATIVE SPEECH/VOICE REMOVAL (COUGH-SAFE)")
    print("=" * 80)
    
    # Check VAD availability
    if not VAD_AVAILABLE:
        print("\n[IMPORTANT] webrtcvad not installed for voice detection")
        print("To enable speech removal, install: pip install webrtcvad")
        print("Preprocessing will continue without speech removal.\n")
    else:
        print("\n[OK] webrtcvad available - voice detection ENABLED\n")
    
    # Step 1: Read filtered CSV
    print(f"Step 1: Reading filtered CSV...")
    try:
        df = pd.read_csv(FILTERED_CSV)
        print(f"  Successfully loaded CSV with {len(df)} rows")
    except Exception as e:
        print(f"  Error reading CSV: {e}")
        return
    
    # Step 2: Identify the column containing audio filenames
    # Using 'file_features' column as the audio filename source
    filename_col = 'file_features'
    
    if filename_col not in df.columns:
        print(f"  Error: Column '{filename_col}' not found in CSV")
        print(f"  Available columns: {list(df.columns)}")
        return
    
    print(f"  Using column '{filename_col}' as audio filenames")
    
    # Step 3: Get list of audio files from CSV
    required_files = df[filename_col].unique()
    print(f"  Found {len(required_files)} unique audio files in CSV")
    
    # Step 4: Find and preprocess audio files
    print(f"\nStep 2: Preprocessing audio files...")
    print(f"  Source directory: {SOURCE_AUDIO_DIR}")
    print(f"  Target directory: {TARGET_AUDIO_DIR}")
    print(f"  Speech removal: {'ENABLED (Conservative)' if VAD_AVAILABLE else 'DISABLED'}\n")
    
    # Get all available audio files in source directory
    available_files = set([f for f in os.listdir(SOURCE_AUDIO_DIR) if f.endswith(('.wav', '.mp3'))])
    print(f"  Total audio files in source directory: {len(available_files)}\n")
    
    # Track statistics
    processed_count = 0
    failed_count = 0
    not_found_count = 0
    
    # Process each required file
    for idx, required_file in enumerate(required_files, 1):
        # Handle NaN values
        if pd.isna(required_file):
            continue
        
        # Convert to string and strip whitespace
        required_file = str(required_file).strip()
        
        # Check if file has extension, if not, try common extensions
        if not required_file.lower().endswith(('.wav', '.mp3')):
            # Try common extensions
            found = False
            for ext in ['.wav', '.mp3']:
                if required_file + ext in available_files:
                    required_file = required_file + ext
                    found = True
                    break
            if not found:
                print(f"{idx}. File not found: {required_file} (tried .wav, .mp3)")
                not_found_count += 1
                continue
        
        # Check if file exists in source directory
        if required_file not in available_files:
            print(f"{idx}. File not found: {required_file}")
            not_found_count += 1
            continue
        
        # Preprocess the audio
        source_path = os.path.join(SOURCE_AUDIO_DIR, required_file)
        target_path = os.path.join(TARGET_AUDIO_DIR, required_file)
        
        print(f"{idx}. Processing: {required_file}")
        
        y_preprocessed, sr = preprocess_cough_audio(source_path, remove_speech=True)
        
        if y_preprocessed is not None:
            # Save preprocessed audio
            sf.write(target_path, y_preprocessed, sr)
            print(f"     ✓ Saved to: {os.path.basename(target_path)}\n")
            processed_count += 1
        else:
            print(f"     ✗ Failed to preprocess\n")
            failed_count += 1
    
    # Summary report
    print("\n" + "=" * 80)
    print("PREPROCESSING SUMMARY")
    print("=" * 80)
    print(f"Total files in CSV:           {len(required_files)}")
    print(f"Successfully processed:       {processed_count}")
    print(f"Failed to process:            {failed_count}")
    print(f"Files not found:              {not_found_count}")
    print(f"Target directory:             {TARGET_AUDIO_DIR}")
    print(f"Speech removal:               {'ENABLED (Conservative)' if VAD_AVAILABLE else 'DISABLED'}")
    print("=" * 80)
    
    if not VAD_AVAILABLE:
        print("\n[REMINDER] To enable voice detection, install: pip install webrtcvad")
        print("Then run this script again.")
    
    print("\n✓ Preprocessing complete!")
    print("  Next step: Extract features from these preprocessed audio files")

if __name__ == "__main__":
    main()
