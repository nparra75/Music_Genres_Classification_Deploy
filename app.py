import streamlit as st
import librosa
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.image import resize
import pickle

# Load models and scaler
@st.cache_resource
def load_resources():
    model1 = load_model('music_genre_model1.keras')
    model2 = load_model('music_genre_model2.keras')
    model3 = load_model('music_genre_model_with_mfcc.keras')  # Modelo MFCC
    with open('scaler.pkl', 'rb') as file:
        scaler = pickle.load(file)
    with open('scaler_new.pkl', 'rb') as file:
        scaler_new = pickle.load(file)  # Scaler para MFCC
    return model1, model2, model3, scaler, scaler_new


model1, model2, model3, scaler, scaler_new = load_resources()


# Define label mapping
label_mapping = {
    0: 'Blues', 1: 'Classical', 2: 'Country', 3: 'Disco', 4: 'Hiphop',
    5: 'Jazz', 6: 'Metal', 7: 'Pop', 8: 'Reggae', 9: 'Rock'
}

def process_audio_for_inference(file_path, sr=22050, chunk_duration=4, overlap_duration=2, target_shape=(150, 150)):
    audio, _ = librosa.load(file_path, sr=sr)
    chunk_samples = int(chunk_duration * sr)
    overlap_samples = int(overlap_duration * sr)
    
    num_chunks = int(np.ceil((len(audio) - chunk_samples) / (chunk_samples - overlap_samples))) + 1
    spectrograms = []

    for i in range(num_chunks):
        start = i * (chunk_samples - overlap_samples)
        end = start + chunk_samples
        chunk = audio[start:end]

        if len(chunk) < chunk_samples:
            continue
        
        # Compute Mel spectrogram
        mel_spectrogram = librosa.feature.melspectrogram(y=chunk, sr=sr, n_mels=128, fmax=8000)
        mel_spectrogram = librosa.power_to_db(mel_spectrogram, ref=np.max)
        
        # Resize to match training dimensions
        mel_spectrogram = resize(np.expand_dims(mel_spectrogram, axis=-1), target_shape)
        spectrograms.append(mel_spectrogram)

    return np.array(spectrograms)

def process_audio_with_mfcc(file_path, duration=30, n_mfcc=60):
    try:
        
        signal, sr = librosa.load(file_path, duration=duration)
        
        mfcc = librosa.feature.mfcc(y=signal, sr=sr, n_mfcc=n_mfcc)
        
        mfcc = np.mean(mfcc.T, axis=0)
        
        mfcc = np.expand_dims(mfcc, axis=-1)
        mfcc = np.expand_dims(mfcc, axis=-1) 
        mfcc = np.expand_dims(mfcc, axis=0)
        return mfcc
    except Exception as e:
        print(f"Error processing file {file_path}: {e}")
        return None


def predict_genre(file_path, model, label_mapping, scaler=None, use_mfcc=False):
    if use_mfcc:
        # Process the file with MFCC
        features = process_audio_with_mfcc(file_path)
        if features is None:
            return "Error: Could not process audio file.", None
        # Apply the MFCC scaler
        features = scaler.transform(features.reshape(1, -1)) if scaler else features
        features = features.reshape(1, 60, 1, 1)
        predictions = model.predict(features)
        predicted_class = predictions.argmax(axis=1)[0]
    else:
        # Process the file with spectrograms
        spectrograms = process_audio_for_inference(file_path)
        if spectrograms.size == 0:
            return "Error: Could not process audio file.", None
        spectrograms_scaled = scaler.transform(spectrograms.reshape(spectrograms.shape[0], -1))
        spectrograms_scaled = spectrograms_scaled.reshape(spectrograms.shape)
        predictions = model.predict(spectrograms_scaled)
        predicted_classes = predictions.argmax(axis=1)
        predicted_class = max(set(predicted_classes), key=list(predicted_classes).count)

    return label_mapping[predicted_class]

validation_accuracies = {
    "Model based on spectograms 1": 0.8487,
    "Model based on spectograms 2": 0.8498,
    "Model based on MFCC": 0.74
}

# Streamlit app UI
st.markdown(
    """
    <style>
    .reportview-container {
        background: linear-gradient(to right, #ff7e5f, #feb47b);
        color: white;
        font-family: 'Helvetica', sans-serif;
    }
    .sidebar .sidebar-content {
        background: #333;
        color: white;
    }
    .css-1d391kg p {
        font-size: 18px;
    }
    </style>
    """,
    unsafe_allow_html=True
)

st.title("🎵 Music Genre Classification App")
st.markdown("This app allows you to classify music genres using three different CNN models. Upload an audio file to get started!")

st.sidebar.header("Model Selection")
st.sidebar.markdown("### Choose a classification model:")
model_choice = st.sidebar.radio(
    "", ("Model based on spectograms 1", "Model based on spectograms 2", "Model based on MFCC")
)

uploaded_file = st.file_uploader("Upload an audio file (WAV/MP3):", type=["wav", "mp3"])

if uploaded_file:
    st.audio(uploaded_file, format="audio/wav")
    with open("temp_audio_file.wav", "wb") as f:
        f.write(uploaded_file.getbuffer())
    st.success("File uploaded successfully!")

if st.button("🎶 Classify Genre"):
    if uploaded_file:
        if model_choice == "Model based on spectograms 1":
            model = model1
            use_mfcc = False
            selected_scaler = scaler
            method = "Spectrogram"
        elif model_choice == "Model based on spectograms 2":
            model = model2
            use_mfcc = False
            selected_scaler = scaler
            method = "Spectrogram"
        else:
            model = model3
            use_mfcc = True
            selected_scaler = scaler_new
            method = "MFCC"

        with st.spinner("Analyzing the audio file..."):
            try:
                predicted_genre = predict_genre("temp_audio_file.wav", model, label_mapping, selected_scaler, use_mfcc)
                st.markdown(f"### Predicted Genre: **:blue[{predicted_genre}]**")

                # Display method used (Spectrogram or MFCC)
                st.markdown(f"### Method Used: **:green[{method}]**")

                # Display precomputed validation accuracy
                val_accuracy = validation_accuracies.get(model_choice, None)
                if val_accuracy is not None:
                    st.markdown(f"### Validation Accuracy: **:orange[{val_accuracy * 100:.2f}%]**")
                else:
                    st.markdown("### Validation Accuracy: Not available for this model.")

            except Exception as e:
                st.error(f"Error during prediction: {e}")

