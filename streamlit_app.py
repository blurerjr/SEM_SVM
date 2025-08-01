import streamlit as st
import librosa
import numpy as np
import joblib
import requests
import io
import os
import soundfile as sf
import matplotlib.pyplot as plt
import textwrap
import tempfile

# --- Configuration ---
# GitHub raw URLs for your saved model components
MODEL_URL = "https://raw.githubusercontent.com/blurerjr/SEM_SVM/refs/heads/master/final_best_model.pkl"
SCALER_URL = "https://raw.githubusercontent.com/blurerjr/SEM_SVM/refs/heads/master/feature_scaler.pkl"
ENCODER_URL = "https://raw.githubusercontent.com/blurerjr/SEM_SVM/refs/heads/master/label_encoder.pkl"

# --- Function to Load Model Components from URL ---
@st.cache_resource
def load_model_components_from_url():
    """Loads the pre-trained model, scaler, and label encoder from GitHub URLs."""
    try:
        # Fetch model
        st.write("Fetching model from GitHub...")
        model_response = requests.get(MODEL_URL)
        model_response.raise_for_status()
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
        st.stop()
    except Exception as e:
        st.error(f"An unexpected error occurred while loading model components: {e}")
        st.stop()

# Load model components once at the start
try:
    model, scaler, label_encoder = load_model_components_from_url()
except Exception as e:
    st.error("Application could not start due to an error loading the model. Please check the logs.")
    st.stop()

# --- Feature Extraction Function ---
def extract_features_for_prediction(audio_data_bytes, sr=22050):
    """
    Extracts audio features (20 MFCCs) from raw audio data bytes.
    Handles different audio formats using an in-memory approach.
    """
    try:
        # Use an in-memory byte stream instead of a temporary file
        audio_stream = io.BytesIO(audio_data_bytes)
        
        # Load the audio data
        y, sr = librosa.load(audio_stream, sr=sr)
        
        mfccs = np.mean(librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20).T, axis=0)
        
        # NOTE: If your model was trained with other features,
        # you must uncomment and concatenate them here.
        
        features = mfccs
        
        return features
    except Exception as e:
        st.error(f"Error during feature extraction: {e}")
        return None

# --- Custom CSS for Waveform Animation and Styling ---
CUSTOM_CSS = textwrap.dedent("""
<style>
    /* General body styling for background */
    body {
        background-color: #f9fafb; /* bg-gray-50 */
    }
    .stApp {
        background-color: #f9fafb; /* Ensure Streamlit app background matches */
    }

    /* Waveform container styling */
    .waveform-container {
        background: linear-gradient(90deg, #3b82f6 0%, #8b5cf6 50%, #ec4899 100%);
        height: 100px;
        position: relative;
        overflow: hidden;
        border-radius: 0.5rem; /* rounded-lg */
        margin-bottom: 1rem; /* mb-4 */
    }
    .wave {
        position: absolute;
        bottom: 0;
        width: 100%;
        height: 100%;
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

    /* Custom button styling for Streamlit buttons */
    .stButton>button {
        background-color: #4f46e5; /* indigo-600 */
        color: white;
        border-radius: 0.5rem; /* rounded-lg */
        padding: 0.75rem 1.5rem; /* px-6 py-3 */
        font-weight: 500; /* font-medium */
        transition: background-color 0.3s ease;
        display: flex;
        align-items: center;
        justify-content: center;
        border: none;
    }
    .stButton>button:hover {
        background-color: #4338ca; /* indigo-700 */
    }
    .stButton>button:disabled {
        background-color: #9ca3af; /* gray-400 */
        cursor: not-allowed;
    }

    /* Specific styling for the "Analyze Emotion" button to match original HTML */
    .stButton>button[key="analyze_button"] { /* Target the specific button by key */
        background-color: #4f46e5; /* indigo-600 */
        color: white;
        font-size: 1.125rem; /* text-lg */
        padding: 0.75rem 1.5rem; /* py-3 px-6 */
        width: 100%; /* w-full */
    }

    /* Styling for the file uploader to match HTML design */
    .stFileUploader > div > div {
        border: 2px dashed #d1d5db; /* border-2 border-dashed border-gray-300 */
        border-radius: 0.5rem; /* rounded-lg */
        padding: 1.5rem; /* p-6 */
        text-align: center;
        display: flex;
        flex-direction: column;
        align-items: center;
        justify-content: center;
    }
    .stFileUploader > div > div > button {
        background-color: #4f46e5; /* indigo-600 */
        color: white;
        border-radius: 0.5rem;
        padding: 0.5rem 1rem; /* px-4 py-2 */
        font-weight: 500;
        transition: background-color 0.3s ease;
    }
    .stFileUploader > div > div > button:hover {
        background-color: #4338ca; /* indigo-700 */
    }
    .stFileUploader > div > div > small { /* "Drag & drop..." text */
        color: #6b7280; /* text-gray-600 */
        margin-bottom: 0.75rem; /* mb-3 */
    }
    .stFileUploader > div > div > div:first-child > div:first-child { /* "Drag & drop..." container */
        display: flex;
        flex-direction: column;
        align-items: center;
    }
    .stFileUploader p { /* Text like "No file uploaded" */
        color: #6b7280; /* text-gray-600 */
    }

    /* Emotion icon styling */
    .emotion-icon-container {
        background-color: #f9fafb; /* bg-gray-50 */
        border-radius: 0.5rem; /* rounded-lg */
        padding: 0.75rem; /* p-3 */
        text-align: center;
        cursor: pointer; /* Indicate clickable */
        transition: transform 0.2s, box-shadow 0.2s;
        height: 100%; /* Ensure consistent height */
        display: flex;
        flex-direction: column;
        justify-content: center;
        align-items: center;
    }
    .emotion-icon-container:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06);
    }
    .emotion-icon-container.selected {
        border: 2px solid #6366f1; /* ring-2 ring-indigo-500 */
    }
    .emotion-icon-container i {
        font-size: 2.25rem; /* text-3xl */
        margin-bottom: 0.25rem; /* mb-1 */
    }
    .emotion-icon-container p {
        font-size: 0.75rem; /* text-xs */
        color: #4b5563; /* text-gray-600 */
    }

    /* Specific emotion colors for icons */
    .bg-gray-100 { background-color: #f3f4f6; } /* Neutral */
    .bg-yellow-50 { background-color: #fffbeb; } /* Happy */
    .bg-blue-50 { background-color: #eff6ff; } /* Sad */
    .bg-red-50 { background-color: #fef2f2; } /* Angry */
    .bg-purple-50 { background-color: #f5f3ff; } /* Fearful */
    .bg-green-50 { background-color: #ecfdf5; } /* Disgust */
    .bg-pink-50 { background-color: #fdf2f8; } /* Surprised */
    .bg-teal-50 { background-color: #f0fdfa; } /* Calm */

    .text-gray-500 { color: #6b7280; }
    .text-yellow-500 { color: #f59e0b; }
    .text-blue-500 { color: #3b82f6; }
    .text-red-500 { color: #ef4444; }
    .text-purple-500 { color: #a855f7; }
    .text-green-500 { color: #22c55e; }
    .text-pink-500 { color: #ec4899; }
    .text-teal-500 { color: #14b8a6; }

    /* Footer styling */
    .st-emotion-cache-1jm6g5h { /* Target the footer container */
        text-align: center;
        color: #6b7280; /* text-gray-500 */
        font-size: 0.875rem; /* text-sm */
        margin-top: 3rem; /* mt-12 */
    }
</style>
<link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css">
""")
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

# --- Streamlit App Layout ---
st.set_page_config(page_title="Speech Emotion Recognition", page_icon="🗣️", layout="wide")

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

    # Audio Input Tabs
    tab1, tab2 = st.tabs(["Record Voice", "Upload Audio File"])

    with tab1:
        st.markdown(textwrap.dedent("""
        <p class="text-gray-600 mb-3">Click the button below to start recording. Once finished, click again to stop.</p>
        <div class="waveform-container">
            <div class="wave"></div>
            <div class="wave"></div>
            <div class="wave"></div>
        </div>
        """), unsafe_allow_html=True)
        
        # Streamlit's native audio input widget for recording
        recorded_audio = st.audio_input("Start Recording", key="recorder_input")
        if recorded_audio:
            st.session_state.audio_data = recorded_audio.read()
            st.audio(recorded_audio, format="audio/wav")
            st.info("Audio recorded successfully! Click 'Analyze Emotion' to process.")

    with tab2:
        st.markdown(textwrap.dedent("""
        <div class="border-2 border-dashed border-gray-300 rounded-lg p-6 text-center">
            <div class="flex flex-col items-center justify-center">
                <i class="fas fa-cloud-upload-alt text-4xl text-indigo-500 mb-3"></i>
                <p class="text-gray-600 mb-3">Drag & drop your audio file here or</p>
                <p class="text-xs text-gray-500 mt-2">Supported formats: WAV, MP3 (max 25MB)</p>
            </div>
        </div>
        """), unsafe_allow_html=True)
        uploaded_file = st.file_uploader("Browse Files", type=["wav", "mp3"], key="uploader_input")
        if uploaded_file:
            st.session_state.audio_data = uploaded_file.read()
            st.audio(uploaded_file, format="audio/wav")
            st.info("Audio file uploaded successfully! Click 'Analyze Emotion' to process.")

    # Analyze Button
    if st.button("Analyze Emotion", use_container_width=True, key="analyze_button"):
        if st.session_state.get('audio_data') is None:
            st.warning("Please record or upload an audio file first.")
        else:
            with st.spinner("Extracting features and predicting emotion..."):
                features = extract_features_for_prediction(st.session_state.audio_data)

                if features is not None:
                    try:
                        features_reshaped = features.reshape(1, -1)
                        scaled_features = scaler.transform(features_reshaped)
                        prediction_encoded = model.predict(scaled_features)
                        predicted_emotion = label_encoder.inverse_transform(prediction_encoded)[0]
                        
                        st.session_state['predicted_emotion'] = predicted_emotion
                        st.session_state['predicted_features'] = features
                        st.session_state['has_prediction'] = True
                        st.rerun()
                    except Exception as e:
                        st.error(f"An error occurred during prediction: {e}")
                else:
                    st.error("Failed to extract features or make a prediction.")
    
    st.markdown("</div>", unsafe_allow_html=True)

with col_results:
    st.markdown(textwrap.dedent("""
    <div class="bg-white rounded-xl shadow-lg p-6">
        <h2 class="text-2xl font-semibold text-gray-800 mb-6">Analysis Results</h2>
    """), unsafe_allow_html=True)

    if 'has_prediction' in st.session_state and st.session_state['has_prediction']:
        predicted_emotion = st.session_state['predicted_emotion']
        predicted_features = st.session_state['predicted_features']

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

        display_emotion = predicted_emotion.replace("male_", "").replace("female_", "").capitalize()
        
        confidence_display = "High"
        
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
        </div>
        """), unsafe_allow_html=True)
        
        emotions = [
            ("neutral", "fas fa-meh", "text-gray-500", "bg-gray-100", "Neutral"),
            ("happy", "fas fa-smile", "text-yellow-500", "bg-yellow-50", "Happy"),
            ("sad", "fas fa-sad-tear", "text-blue-500", "bg-blue-50", "Sad"),
            ("angry", "fas fa-angry", "text-red-500", "bg-red-50", "Angry"),
            ("fear", "fas fa-grimace", "text-purple-500", "bg-purple-50", "Fearful"),
            ("disgust", "fas fa-grimace", "text-green-500", "bg-green-50", "Disgust"),
            ("surprise", "fas fa-surprise", "text-pink-500", "bg-pink-50", "Surprised"),
            ("calm", "fas fa-spa", "text-teal-500", "bg-teal-50", "Calm"),
        ]

        emotion_icons_html = []
        for emotion_key, icon, color, bg, label in emotions:
            selected_class = "selected" if emotion_key in predicted_emotion.lower() else ""
            html_content = f"""
            <div class="emotion-icon-container {selected_class}">
                <div class="{bg} rounded-lg p-3 text-center">
                    <i class="{icon} text-3xl {color} mb-1"></i>
                    <p class="text-xs text-gray-600">{label}</p>
                </div>
            </div>
            """
            emotion_icons_html.append(html_content)
        
        final_icons_html = textwrap.dedent(f"""
        <div class="grid grid-cols-4 gap-3 mt-6">
            {"".join(emotion_icons_html)}
        </div>
        """)
        
        st.markdown(final_icons_html, unsafe_allow_html=True)

        st.subheader("📊 Audio Features (MFCCs)")
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.bar(range(len(predicted_features)), predicted_features, color='skyblue')
        ax.set_xlabel("MFCC Coefficient Index")
        ax.set_ylabel("Value")
        ax.set_title("Extracted MFCC Features")
        st.pyplot(fig)
        plt.close(fig)

    else:
        st.markdown(textwrap.dedent("""
        <div id="resultsPlaceholder" class="text-center py-12">
            <i class="fas fa-wave-square text-5xl text-gray-300 mb-4"></i>
            <p class="text-gray-500">Record or upload audio to analyze emotions</p>
        </div>
        """), unsafe_allow_html=True)

    st.markdown("</div>", unsafe_allow_html=True)

st.markdown("---")

st.subheader("📈 Help Improve the Model")
st.markdown("""
Your feedback is valuable for making this model even better!
In the future, we plan to add features here that allow you to:
* Rate the accuracy of the prediction.
* Provide correct labels for misclassified audio.
* Contribute new audio samples for training.
""")

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
