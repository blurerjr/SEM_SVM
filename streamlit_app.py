import streamlit as st
import librosa
import numpy as np
import joblib
import requests
import io
import os
import soundfile as sf # Required by librosa for writing audio files

# --- Configuration ---
# GitHub raw URLs for your saved model components
# IMPORTANT: Ensure these URLs are correct and publicly accessible
MODEL_URL = "https://raw.githubusercontent.com/blurerjr/SEM_SVM/refs/heads/master/final_best_model.pkl"
SCALER_URL = "https://raw.githubusercontent.com/blurerjr/SEM_SVM/refs/heads/master/feature_scaler.pkl"
ENCODER_URL = "https://raw.githubusercontent.com/blurerjr/SEM_SVM/refs/heads/master/label_encoder.pkl"

# Temporary file path for saving recorded audio
TEMP_AUDIO_FILE = "recorded_audio.wav"

# --- Function to Load Model Components from URL ---
@st.cache_resource # Cache the loading of these heavy resources
def load_model_components_from_url():
    """Loads the pre-trained model, scaler, and label encoder from GitHub URLs."""
    try:
        # Fetch model
        st.write("Fetching model from GitHub...")
        model_response = requests.get(MODEL_URL)
        model_response.raise_for_status() # Raise an exception for HTTP errors
        model = joblib.load(io.BytesIO(model_response.content))
        st.success("Model loaded.")

        # Fetch scaler
        st.write("Fetching scaler from GitHub...")
        scaler_response = requests.get(SCALER_URL)
        scaler_response.raise_for_status()
        scaler = joblib.load(io.BytesIO(scaler_response.content))
        st.success("Scaler loaded.")

        # Fetch label encoder
        st.write("Fetching label encoder from GitHub...")
        encoder_response = requests.get(ENCODER_URL)
        encoder_response.raise_for_status()
        label_encoder = joblib.load(io.BytesIO(encoder_response.content))
        st.success("Label Encoder loaded.")

        return model, scaler, label_encoder
    except requests.exceptions.RequestException as e:
        st.error(f"Failed to fetch model components from GitHub. Please check URLs and internet connection: {e}")
        st.stop() # Stop the app if crucial files can't be loaded
    except Exception as e:
        st.error(f"An unexpected error occurred while loading model components: {e}")
        st.stop()

# Load model components once at the start
model, scaler, label_encoder = load_model_components_from_url()

# --- Feature Extraction Function (MUST match your training process) ---
# This function must extract the exact same features (type, number, order)
# as your 'features.csv' was created with.
# Based on previous discussions, we're assuming 20 MFCCs.
def extract_features_for_prediction(audio_file_path, sr=22050):
    """
    Extracts audio features (20 MFCCs) from a given audio file.
    Matches the assumed feature extraction from your training phase.
    """
    try:
        y, sr = librosa.load(audio_file_path, sr=sr)
        
        # Ensure n_mfcc matches the number of features in your training data (20 in your CSV)
        mfccs = np.mean(librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20).T, axis=0)
        
        # If your training used other features (chroma, ZCR, etc.) concatenated,
        # you MUST uncomment and add them here in the exact same order:
        # stft = np.abs(librosa.stft(y))
        # chroma = np.mean(librosa.feature.chroma_stft(S=stft, sr=sr).T, axis=0)
        # mel_spec = np.mean(librosa.feature.melspectrogram(y=y, sr=sr).T, axis=0)
        # zcr = np.mean(librosa.feature.zero_crossing_rate(y=y).T, axis=0)
        # spectral_centroid = np.mean(librosa.feature.spectral_centroid(y=y, sr=sr).T, axis=0)
        # spectral_rolloff = np.mean(librosa.feature.spectral_rolloff(y=y, sr=sr).T, axis=0)
        # rms = np.mean(librosa.feature.rms(y=y).T, axis=0)
        # features = np.hstack((mfccs, chroma, mel_spec, zcr, spectral_centroid, spectral_rolloff, rms))
        
        features = mfccs # Sticking to 20 MFCCs based on your CSV's 20 columns
        
        return features
    except Exception as e:
        st.error(f"Error during feature extraction: {e}")
        return None

# --- Streamlit App Layout ---
st.set_page_config(page_title="Speech Emotion Recognition", page_icon="🗣️")

st.title("🗣️ Speech Emotion Recognition (SER)")
st.markdown("""
Welcome to the Speech Emotion Recognition app!
Record a short speech segment using your microphone, and I'll predict the emotion.
""")

st.warning("Please allow microphone access in your browser when prompted.")

# Streamlit's native audio input widget for recording
uploaded_audio = st.audio_input("Record your speech here", type="wav")

if uploaded_audio is not None:
    # Display the recorded audio for playback
    st.markdown("---")
    st.subheader("Recorded Audio:")
    st.audio(uploaded_audio, format="audio/wav")

    # Save the uploaded file to a temporary .wav file for Librosa processing
    with open(TEMP_AUDIO_FILE, "wb") as f:
        f.write(uploaded_audio.read())
    
    st.info("Audio recorded successfully! Click 'Predict Emotion' to analyze.")

    if st.button("Predict Emotion"):
        with st.spinner("Extracting features and predicting emotion..."):
            features = extract_features_for_prediction(TEMP_AUDIO_FILE)

            if features is not None:
                # Reshape features for the scaler (expects 2D array: [n_samples, n_features])
                features_reshaped = features.reshape(1, -1)
                
                # Scale the features using the loaded scaler
                scaled_features = scaler.transform(features_reshaped)
                
                # Make prediction with the loaded model
                prediction_encoded = model.predict(scaled_features)
                
                # Decode the numerical prediction back to the original emotion label string
                predicted_emotion = label_encoder.inverse_transform(prediction_encoded)[0]
                
                st.success(f"Predicted Emotion: **{predicted_emotion.upper()}**")
            else:
                st.error("Failed to extract features or make a prediction.")
    
    # Clean up the temporary audio file
    if os.path.exists(TEMP_AUDIO_FILE):
        os.remove(TEMP_AUDIO_FILE)

else:
    st.info("Start by clicking the record button above and speaking into your microphone.")

st.markdown("---")
st.markdown("Model trained on RAVDESS dataset using SVM. Developed by blurerjr.")

