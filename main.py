import librosa
import pandas as pd
import os
import numpy as np
import streamlit as st
import joblib

def extract_audio_features(file_path):
    """Extracts the specified audio features from a file."""
    y, sr = librosa.load(file_path)

    features = {
        'filename': os.path.basename(file_path),
        'length': librosa.get_duration(y=y, sr=sr),
        'chroma_stft_mean': np.mean(librosa.feature.chroma_stft(y=y, sr=sr)),
        'chroma_stft_var': np.var(librosa.feature.chroma_stft(y=y, sr=sr)),
        'rms_mean': np.mean(librosa.feature.rms(y=y)),
        'rms_var': np.var(librosa.feature.rms(y=y)),
        'spectral_centroid_mean': np.mean(librosa.feature.spectral_centroid(y=y, sr=sr)),
        'spectral_centroid_var': np.var(librosa.feature.spectral_centroid(y=y, sr=sr)),
        'spectral_bandwidth_mean': np.mean(librosa.feature.spectral_bandwidth(y=y, sr=sr)),
        'spectral_bandwidth_var': np.var(librosa.feature.spectral_bandwidth(y=y, sr=sr)),
        'rolloff_mean': np.mean(librosa.feature.spectral_rolloff(y=y, sr=sr)),
        'rolloff_var': np.var(librosa.feature.spectral_rolloff(y=y, sr=sr)),
        'zero_crossing_rate_mean': np.mean(librosa.feature.zero_crossing_rate(y)),
        'zero_crossing_rate_var': np.var(librosa.feature.zero_crossing_rate(y)),
        'harmony_mean': np.mean(librosa.effects.harmonic(y)),
        'harmony_var': np.var(librosa.effects.harmonic(y)),
        'perceptr_mean': np.mean(librosa.effects.percussive(y)),
        'perceptr_var': np.var(librosa.effects.percussive(y)),
        'tempo': librosa.beat.tempo(y=y, sr=sr)[0]
    }

    # Extract MFCCs 
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20)
    for i in range(1, 21):
        features[f'mfcc{i}_mean'] = np.mean(mfcc[i-1])
        features[f'mfcc{i}_var'] = np.var(mfcc[i-1])

    return features

def main():
    st.title("Audio Feature Extractor & Genre Predictor")
    
    model = joblib.load('genre_classifier_model.pkl')

    uploaded_file = st.file_uploader("Choose a .wav audio file", type="wav")

    if uploaded_file is not None:

        temp_filepath = os.path.join("temp", uploaded_file.name)


        with open(temp_filepath, "wb") as f:
            f.write(uploaded_file.read())


        features = extract_audio_features(temp_filepath)


        st.subheader("Extracted Features:")
        st.write(pd.DataFrame([features])) 


        df = pd.DataFrame([features])


        csv = df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="Download Features as CSV",
            data=csv,
            file_name='extracted_features.csv',
            mime='text/csv',
        )


        os.remove(temp_filepath)
                
        numeric_features = {
            key: float(value) for key, value in features.items() if key != 'filename'
        }

        features_for_prediction = np.array(list(numeric_features.values())).reshape(1, -1)
        predicted_genre = model.predict(features_for_prediction)[0]

        st.subheader("Predicted Genre:")
        st.write(predicted_genre)

        try:
            existing_df = pd.read_csv("Data/features_30_sec.csv")
        except FileNotFoundError:
            existing_df = pd.DataFrame() 


        df = pd.concat([existing_df, pd.DataFrame([features])], ignore_index=True)
        df.to_csv("Data/features_30_sec.csv", index=False)

if __name__ == "__main__":
    main()
