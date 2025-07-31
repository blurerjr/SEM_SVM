import streamlit as st
import librosa
import numpy as np
import joblib
import requests
import io
import os
import soundfile as sf # Required by librosa for writing audio files
import matplotlib.pyplot as plt # For plotting features
import textwrap # For dedenting HTML strings

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

# --- Custom CSS for Waveform Animation (from your HTML) ---
# This will be embedded using st.markdown(unsafe_allow_html=True)
CUSTOM_CSS = textwrap.dedent("""
<style>
    .waveform-container {
        background: linear-gradient(90deg, #3b82f6 0%, #8b5cf6 50%, #ec4899 100%);
        height: 100px; /* Reduced height for Streamlit compatibility */
        position: relative;
        overflow: hidden;
        border-radius: 0.5rem; /* rounded-lg */
        margin-bottom: 1rem; /* mb-4 */
    }
    .wave {
        position: absolute;
        bottom: 0;
        width: 100%;
        height: 100%; /* Fill container */
        background: url('data:image/svg+xml;utf8,<svg viewBox="0 0 1200 120" xmlns="http://www.w3.org/2000/svg"><path d="M0,0V46.29c47.79,22.2,103.59,32.17,158,28,70.36-5.37,136.33-33.31,206.8-37.5C438.64,32.43,512.34,53.67,583,72.05c69.27,18,138.3,24.88,209.4,13.08,36.15-6,69.85-17.84,104.45-29.34C989.49,25,1113-14.29,1200,52.47V0Z" opacity=".25" fill="white"/><path d="M0,0V15.81C13,36.92,27.64,56.86,47.69,72.05,99.41,111.27,165,111,224.58,91.58c31.15-10.15,60.09-26.07,89.67-39.8,40.92-19,84.73-46,130.83-49.67,36.26-2.85,70.9,9.42,98.6,31.56,31.77,25.39,62.32,62,103.63,73,40.44,10.79,81.35-6.69,119.13-24.28s75.16-39,116.92-43.05c59.73-5.85,113.28,22.88,168.9,38.84,30.2,8.66,59,6.17,87.09-7.5,22.43-10.89,48-26.93,60.65-49.24V0Z" opacity=".5" fill="white"/><path d="M0,0V5.63C149.93,59,314.09,71.32,475.83,42.57c43-7.64,84.23-20.12,127.61-26.46,59-8.63,112.48,12.24,165.56,35.4C827.93,77.22,886,95.24,951.2,90c86.53-7,172.46-45.71,248.8-84.81V0Z" fill="white"/>');
        background-repeat: repeat-x;
        background-size: 1200px 100px;
        animation: wave 12s linear infinite;
    }
    .wave:nth-child(2) {
        animation: wave 8s linear infinite reverse;
        opacity: 0.5;
    }
    .wave:nth-child(3) {
        animation: wave 10s linear infinite;
        opacity: 0.7;
    }
    @keyframes wave {
        0% { background-position-x: 0; }
        100% { background-position-x: 1200px; }
    }
    .pulse-indicator {
        animation: pulse 2s cubic-bezier(0.4, 0, 0.6, 1) infinite;
    }
    @keyframes pulse {
        0%, 100% { opacity: 1; }
        50% { opacity: 0.5; }
    }
</style>
""")
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

# --- Streamlit App Layout ---
st.set_page_config(page_title="Speech Emotion Recognition", page_icon="🗣️", layout="wide")

# Diagnostic: Print Streamlit version
st.sidebar.write(f"Streamlit Version: {st.__version__}")

st.markdown(textwrap.dedent("""
<header class="mb-10 text-center">
    <h1 class="text-4xl font-bold text-indigo-700 mb-2">Speech Emotion Recognition</h1>
    <p class="text-lg text-gray-600">Analyze emotions in speech using machine learning</p>
</header>
"""), unsafe_allow_html=True)

st.warning("Please allow microphone access in your browser when prompted.")

# Main content area with two columns
col_input, col_results = st.columns(2)

with col_input:
    st.markdown(textwrap.dedent("""
    <div class="bg-white rounded-xl shadow-lg p-6">
        <h2 class="text-2xl font-semibold text-gray-800 mb-6">Record or Upload Audio</h2>
    """), unsafe_allow_html=True)

    # Recording Section (using Streamlit's native audio_input)
    st.subheader("🎙️ Record Your Voice")
    st.markdown(textwrap.dedent("""
    <p class="text-gray-600 mb-3">Click the button below to start recording. Once finished, click again to stop.</p>
    <div class="waveform-container">
        <div class="wave"></div>
        <div class="wave"></div>
        <div class="wave"></div>
    </div>
    """), unsafe_allow_html=True)

    uploaded_audio = st.audio_input("Start Recording")

    if uploaded_audio is not None:
        st.markdown("---")
        st.subheader("Recorded Audio Playback:")
        st.audio(uploaded_audio, format="audio/wav")

        # Save the uploaded file to a temporary .wav file for Librosa processing
        with open(TEMP_AUDIO_FILE, "wb") as f:
            f.write(uploaded_audio.read())
        
        st.info("Audio recorded successfully! Click 'Predict Emotion' to analyze.")

        if st.button("Predict Emotion", use_container_width=True):
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
                    
                    st.session_state['predicted_emotion'] = predicted_emotion
                    st.session_state['predicted_features'] = features
                    st.session_state['has_prediction'] = True
                    st.rerun() # Rerun to update the results column
                else:
                    st.error("Failed to extract features or make a prediction.")
        
        # Clean up the temporary audio file
        if os.path.exists(TEMP_AUDIO_FILE):
            os.remove(TEMP_AUDIO_FILE)

    else:
        st.info("Click 'Start Recording' to begin capturing your voice.")

    st.markdown("</div>", unsafe_allow_html=True) # Close the bg-white container

with col_results:
    st.markdown(textwrap.dedent("""
    <div class="bg-white rounded-xl shadow-lg p-6">
        <h2 class="text-2xl font-semibold text-gray-800 mb-6">Analysis Results</h2>
    """), unsafe_allow_html=True)

    if 'has_prediction' in st.session_state and st.session_state['has_prediction']:
        predicted_emotion = st.session_state['predicted_emotion']
        predicted_features = st.session_state['predicted_features']

        # --- Gender Prediction (Simplified) ---
        # Assuming gender is part of your emotion label (e.g., 'male_calm', 'female_happy')
        gender = "Male" if "male" in predicted_emotion else "Female"
        gender_icon = "fas fa-mars text-blue-600" if gender == "Male" else "fas fa-venus text-pink-600"
        
        st.markdown(textwrap.dedent(f"""
        <div class="mb-8">
            <h3 class="text-lg font-medium text-gray-700 mb-4">Speaker Gender</h3>
            <div class="flex items-center">
                <div class="w-16 h-16 rounded-full bg-indigo-100 flex items-center justify-center mr-4">
                    <i class="{gender_icon} text-2xl"></i>
                </div>
                <div>
                    <p class="text-sm text-gray-500">Predicted gender</p>
                    <p class="text-xl font-semibold text-gray-800">{gender}</p>
                    <p class="text-sm text-gray-500">Confidence: N/A (single prediction)</p>
                </div>
            </div>
        </div>
        """), unsafe_allow_html=True)

        # --- Emotion Prediction ---
        # Clean up emotion name for display (e.g., 'male_calm' -> 'Calm')
        display_emotion = predicted_emotion.replace("male_", "").replace("female_", "").capitalize()
        
        # Map emotion to a confidence color/bar (simplified, as we don't have probabilities directly from SVM predict)
        # For a real application, you'd use model.predict_proba() if available and apply softmax
        # Here, we'll use a fixed high confidence for display since SVM's .predict gives a single class
        confidence_display = "High" # Placeholder for display
        
        st.markdown(textwrap.dedent(f"""
        <div class="mb-8">
            <h3 class="text-lg font-medium text-gray-700 mb-4">Emotion Detection</h3>
            <div class="bg-gray-50 rounded-lg p-4">
                <div class="flex justify-between items-center mb-2">
                    <span class="font-medium text-gray-800">{display_emotion}</span>
                    <span class="text-sm text-gray-500">Confidence: {confidence_display}</span>
                </div>
                <div class="w-full bg-gray-200 rounded-full h-2.5">
                    <div class="bg-gradient-to-r from-blue-500 to-purple-600 h-2.5 rounded-full" style="width: 90%"></div>
                </div>
            </div>
            
            <div class="grid grid-cols-4 gap-3 mt-6">
                <div class="emotion-icon">
                    <div class="bg-gray-100 rounded-lg p-3 text-center {"ring-2 ring-indigo-500" if "neutral" in predicted_emotion else ""}">
                        <i class="fas fa-meh text-3xl text-gray-500 mb-1"></i>
                        <p class="text-xs text-gray-600">Neutral</p>
                    </div>
                </div>
                <div class="emotion-icon">
                    <div class="bg-yellow-50 rounded-lg p-3 text-center {"ring-2 ring-indigo-500" if "happy" in predicted_emotion else ""}">
                        <i class="fas fa-smile text-3xl text-yellow-500 mb-1"></i>
                        <p class="text-xs text-gray-600">Happy</p>
                    </div>
                </div>
                <div class="emotion-icon">
                    <div class="bg-blue-50 rounded-lg p-3 text-center {"ring-2 ring-indigo-500" if "sad" in predicted_emotion else ""}">
                        <i class="fas fa-sad-tear text-3xl text-blue-500 mb-1"></i>
                        <p class="text-xs text-gray-600">Sad</p>
                    </div>
                </div>
                <div class="emotion-icon">
                    <div class="bg-red-50 rounded-lg p-3 text-center {"ring-2 ring-indigo-500" if "angry" in predicted_emotion else ""}">
                        <i class="fas fa-angry text-3xl text-red-500 mb-1"></i>
                        <p class="text-xs text-gray-600">Angry</p>
                    </div>
                </div>
                 <div class="emotion-icon">
                    <div class="bg-purple-50 rounded-lg p-3 text-center {"ring-2 ring-indigo-500" if "fear" in predicted_emotion else ""}">
                        <i class="fas fa-grimace text-3xl text-purple-500 mb-1"></i>
                        <p class="text-xs text-gray-600">Fearful</p>
                    </div>
                </div>
                <div class="emotion-icon">
                    <div class="bg-green-50 rounded-lg p-3 text-center {"ring-2 ring-indigo-500" if "disgust" in predicted_emotion else ""}">
                        <i class="fas fa-grimace text-3xl text-green-500 mb-1"></i>
                        <p class="text-xs text-gray-600">Disgust</p>
                    </div>
                </div>
                <div class="emotion-icon">
                    <div class="bg-pink-50 rounded-lg p-3 text-center {"ring-2 ring-indigo-500" if "surprise" in predicted_emotion else ""}">
                        <i class="fas fa-surprise text-3xl text-pink-500 mb-1"></i>
                        <p class="text-xs text-gray-600">Surprised</p>
                    </div>
                </div>
                <div class="emotion-icon">
                    <div class="bg-teal-50 rounded-lg p-3 text-center {"ring-2 ring-indigo-500" if "calm" in predicted_emotion else ""}">
                        <i class="fas fa-spa text-3xl text-teal-500 mb-1"></i>
                        <p class="text-xs text-gray-600">Calm</p>
                    </div>
                </div>
            </div>
        </div>
        """), unsafe_allow_html=True)

        # --- Feature Visualization (using Matplotlib/Streamlit's native plot) ---
        st.subheader("📊 Audio Features (MFCCs)")
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.bar(range(len(predicted_features)), predicted_features, color='skyblue')
        ax.set_xlabel("MFCC Coefficient Index")
        ax.set_ylabel("Value")
        ax.set_title("Extracted MFCC Features")
        st.pyplot(fig)
        plt.close(fig) # Close the figure to prevent it from displaying multiple times

    else:
        st.markdown(textwrap.dedent("""
        <div id="resultsPlaceholder" class="text-center py-12">
            <i class="fas fa-wave-square text-5xl text-gray-300 mb-4"></i>
            <p class="text-gray-500">Record or upload audio to analyze emotions</p>
        </div>
        """), unsafe_allow_html=True)

    st.markdown("</div>", unsafe_allow_html=True) # Close the bg-white container

st.markdown("---")

# --- Model Improvement Section ---
st.subheader("📈 Help Improve the Model")
st.markdown("""
Your feedback is valuable for making this model even better!
In the future, we plan to add features here that allow you to:
* Rate the accuracy of the prediction.
* Provide correct labels for misclassified audio.
* Contribute new audio samples for training.
""")

# Example of a simple feedback mechanism (not connected to model training yet)
with st.expander("Give Feedback (Optional)"):
    st.write("Was the prediction accurate?")
    col1_fb, col2_fb = st.columns(2)
    with col1_fb:
        if st.button("👍 Yes, Accurate", key="feedback_yes"):
            st.success("Thank you for your feedback!")
    with col2_fb:
        if st.button("👎 No, Inaccurate", key="feedback_no"):
            st.warning("Sorry about that! We'll use your feedback to improve.")
            feedback_text = st.text_area("What was the correct emotion or issue?", key="feedback_text")
            if st.button("Submit detailed feedback", key="submit_feedback"):
                st.info("Feedback submitted. (This would typically save to a database for future model retraining)")


st.markdown("---")
st.markdown(textwrap.dedent("""
<footer class="mt-12 text-center text-gray-500 text-sm">
    <p>Speech Emotion Recognition using Librosa for feature extraction and SVM for classification</p>
    <p class="mt-1">Trained on the RAVDESS dataset. Developed by blurerjr.</p>
</footer>
"""), unsafe_allow_html=True)
