# app_pd_classification.py

import streamlit as st
import numpy as np
import joblib
import parselmouth
import librosa
import nolds
import antropy as ant
import tensorflow as tf
from tensorflow.keras.layers import Layer
import tempfile
import os

# Custom Attention Layer
class Attention(Layer):
    def __init__(self, **kwargs):
        super(Attention, self).__init__(**kwargs)

    def build(self, input_shape):
        self.W = self.add_weight(
            shape=(input_shape[-1], 1),
            initializer="glorot_uniform",
            trainable=True
        )
        self.b = self.add_weight(
            shape=(input_shape[1], 1),
            initializer="zeros",
            trainable=True
        )
        super(Attention, self).build(input_shape)

    def call(self, inputs):
        e = tf.nn.tanh(tf.tensordot(inputs, self.W, axes=1) + self.b)
        alpha = tf.nn.softmax(e, axis=1)
        context = tf.reduce_sum(alpha * inputs, axis=1)
        return context

@st.cache_resource
def load_all():
    model = tf.keras.models.load_model(
        "pd_lstm_attention_model.keras",
        custom_objects={"Attention": Attention},
        compile=False
    )
    scaler = joblib.load("scaler.pkl")
    feature_columns = joblib.load("feature_columns.pkl")
    return model, scaler, feature_columns


model, scaler, feature_columns = load_all()

# UI
st.title("Parkinson's Disease Classification (LSTM + Attention)")

st.sidebar.header("Patient Information")

age = st.sidebar.number_input("Age", 1, 120, 60)

sex_option = st.sidebar.selectbox("Sex", ["Male", "Female"])
sex = 0 if sex_option == "Male" else 1

st.sidebar.header("Upload Patient Recordings (10 required)")
uploaded_files = st.sidebar.file_uploader(
    "Upload exactly 10 WAV files",
    type=["wav"],
    accept_multiple_files=True
)

predict_button = st.button("Run Prediction")


# Feature Extraction
def extract_features(file_path):

    jitter_abs = jitter_rap = jitter_ddp = shimmer_dda = shimmer_apq5 = np.nan
    hnr = nhr = dfa = rpde = ppe = np.nan

    snd = parselmouth.Sound(file_path)

    try:
        harmonicity = parselmouth.praat.call(
            snd, "To Harmonicity (cc)", 0.01, 75, 0.1, 1.0
        )
        hnr_val = parselmouth.praat.call(harmonicity, "Get mean", 0, 0)
        if hnr_val and hnr_val != 0:
            hnr = hnr_val
            nhr = 1 / hnr
    except:
        pass

    try:
        pointProcess = parselmouth.praat.call(
            snd, "To PointProcess (periodic, cc)", 75, 500
        )
        num_points = parselmouth.praat.call(pointProcess, "Get number of points")

        if num_points > 10:
            jitter_abs = parselmouth.praat.call(
                pointProcess, "Get jitter (local, absolute)",
                0, 0, 0.0001, 0.02, 1.3
            )
            jitter_rap = parselmouth.praat.call(
                pointProcess, "Get jitter (rap)",
                0, 0, 0.0001, 0.02, 1.3
            )
            jitter_ddp = parselmouth.praat.call(
                pointProcess, "Get jitter (ddp)",
                0, 0, 0.0001, 0.02, 1.3
            )
            shimmer_dda = parselmouth.praat.call(
                [snd, pointProcess], "Get shimmer (dda)",
                0, 0, 0.0001, 0.02, 1.3, 1.6
            )
            shimmer_apq5 = parselmouth.praat.call(
                [snd, pointProcess], "Get shimmer (apq5)",
                0, 0, 0.0001, 0.02, 1.3, 1.6
            )
    except:
        pass

    y, sr = librosa.load(file_path, sr=None)

    if len(y) > 10:
        try:
            dfa = nolds.dfa(y, overlap=True)
        except:
            pass
        try:
            rpde = ant.sample_entropy(y)
        except:
            pass
        try:
            pitch = snd.to_pitch()
            f0 = pitch.selected_array['frequency']
            voiced_f0 = f0[f0 > 0]
            ppe = np.std(voiced_f0) if len(voiced_f0) > 0 else np.nan
        except:
            pass

    record = {
        "DFA": dfa,
        "RPDE": rpde,
        "HNR": hnr,
        "NHR": nhr,
        "PPE": ppe,
        "Jitter_Abs": jitter_abs,
        "Jitter_RAP": jitter_rap,
        "Jitter_DDP": jitter_ddp,
        "Shimmer_DDA": shimmer_dda,
        "Shimmer_APQ5": shimmer_apq5,
        "age": age,
        "sex": sex
    }

    return [record.get(col, 0.0) for col in feature_columns]


# Prediction
if predict_button:

    if not uploaded_files or len(uploaded_files) != 10:
        st.warning("Please upload exactly 10 recordings.")
    else:

        progress_bar = st.progress(0)
        status_text = st.empty()
        feature_matrix = []

        for i, uploaded_file in enumerate(uploaded_files):

            status_text.text(f"Processing file {i+1}/10")

            with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
                tmp.write(uploaded_file.read())
                tmp_path = tmp.name

            features = extract_features(tmp_path)
            feature_matrix.append(features)
            os.remove(tmp_path)

            progress_bar.progress((i + 1) / 10)

        X_input = np.array(feature_matrix)
        X_input = np.nan_to_num(X_input, nan=0.0)

        X_input_scaled = scaler.transform(X_input)
        X_input_scaled = np.expand_dims(X_input_scaled, axis=0)

        with st.spinner("Running LSTM model..."):
            pred_prob = model.predict(X_input_scaled).ravel()[0]

        pred_class = "PD" if pred_prob >= 0.5 else "Healthy"

        st.success(f"Prediction: {pred_class}")
        st.write(f"Probability of PD: {pred_prob:.4f}")

        progress_bar.empty()
        status_text.empty()

st.divider()
st.markdown("""
---
**Prepared by:** Haidar Al-Hussein  
**Model:** BiLSTM + Attention Network  
**Application:** Early Parkinson’s Disease Screening  
""")