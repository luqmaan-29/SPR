# ==============================================
# Streamlit Speech-to-Text (Live Mic + Google + Vosk + Whisper + WER)
# ==============================================
import streamlit as st
import sounddevice as sd
import soundfile as sf
import tempfile
import subprocess
import json
import pandas as pd
import speech_recognition as sr
from vosk import Model, KaldiRecognizer
import os
from jiwer import wer
import whisper

# ----------------------------
# Load Models
# ----------------------------

# Vosk model path
VOSK_MODEL_PATH = "vosk-model-small-en-us-0.15"  # download this model first
if not os.path.exists(VOSK_MODEL_PATH):
    st.warning("⚠️ Please download Vosk model: https://alphacephei.com/vosk/models "
               "and unzip it as 'vosk-model-small-en-us-0.15'")
else:
    vosk_model = Model(VOSK_MODEL_PATH)

# Whisper model
st.info("Loading Whisper model (may take a while)...")
whisper_model = whisper.load_model("small")  # options: tiny, base, small, medium, large

# ----------------------------
# Record Audio from Mic
# ----------------------------
def record_audio(duration=5, fs=16000):
    st.info(f"🎤 Speak something! Recording audio for {duration} seconds...")
    audio = sd.rec(int(duration * fs), samplerate=fs, channels=1, dtype='int16')
    sd.wait()
    tmp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
    sf.write(tmp_file.name, audio, fs)
    st.success(f"Recording saved!")
    return tmp_file.name

# ----------------------------
# Google Speech API
# ----------------------------
def recognize_google(filename):
    r = sr.Recognizer()
    with sr.AudioFile(filename) as source:
        audio_data = r.record(source)
        try:
            st.info("🔍 Recognizing with Google API...")
            text = r.recognize_google(audio_data)
            st.success("✅ Google Speech successfully converted to text!")
            return text
        except sr.UnknownValueError:
            return "⚠️ Could not understand audio. Speak clearly."
        except sr.RequestError:
            return "⚠️ Google API unavailable. Check your internet."

# ----------------------------
# Vosk Recognition (Offline)
# ----------------------------
def recognize_vosk(filename):
    try:
        st.info("🔍 Recognizing with Vosk (offline)...")
        process = subprocess.Popen(
            ["ffmpeg", "-loglevel", "quiet", "-i", filename, "-ar", "16000", "-ac", "1", "-f", "s16le", "-"],
            stdout=subprocess.PIPE
        )
        rec = KaldiRecognizer(vosk_model, 16000)
        result_text = ""
        while True:
            data = process.stdout.read(4000)
            if len(data) == 0:
                break
            if rec.AcceptWaveform(data):
                res = json.loads(rec.Result())
                result_text += " " + res.get("text", "")
        final_res = json.loads(rec.FinalResult())
        result_text += " " + final_res.get("text", "")
        if result_text.strip() == "":
            return "⚠️ Could not understand audio. Speak clearly."
        st.success("✅ Vosk Speech successfully converted to text!")
        return result_text.strip()
    except Exception as e:
        return f"⚠️ Vosk error: {str(e)}"

# ----------------------------
# Whisper Recognition (Offline)
# ----------------------------
def recognize_whisper(filename):
    st.info("🔍 Recognizing with Whisper (offline)...")
    try:
        result = whisper_model.transcribe(filename)
        text = result["text"].strip()
        if text == "":
            return "⚠️ Could not understand audio. Speak clearly."
        st.success("✅ Whisper successfully converted to text!")
        return text
    except Exception as e:
        return f"⚠️ Whisper error: {str(e)}"

# ----------------------------
# Compute WER and Accuracy
# ----------------------------
def compute_accuracy(pred_text, ground_truth):
    try:
        error = wer(ground_truth.lower(), pred_text.lower())
        accuracy = max(0, (1 - error) * 100)  # ensure accuracy >= 0
        return round(accuracy, 2)
    except:
        return 0

# ----------------------------
# Compare Methods with Accuracy
# ----------------------------
def compare_methods(filename, ground_truth):
    google_text = recognize_google(filename)
    vosk_text = recognize_vosk(filename)
    whisper_text = recognize_whisper(filename)
    
    results = {
        "Method": ["Google API", "Vosk (Offline)", "Whisper (Offline)"],
        "Output": [google_text, vosk_text, whisper_text],
        "WER": [
            round(wer(ground_truth.lower(), google_text.lower()), 2),
            round(wer(ground_truth.lower(), vosk_text.lower()), 2),
            round(wer(ground_truth.lower(), whisper_text.lower()), 2)
        ],
        "Accuracy (%)": [
            compute_accuracy(google_text, ground_truth),
            compute_accuracy(vosk_text, ground_truth),
            compute_accuracy(whisper_text, ground_truth)
        ]
    }
    
    df = pd.DataFrame(results)
    return df

# ----------------------------
# Streamlit UI
# ----------------------------
st.title("🎙️ Speech-to-Text Comparison with Google, Vosk & Whisper")
st.write("Record your voice and see text conversion using Google API, Vosk, and Whisper, with WER and Accuracy metrics.")

duration = st.slider("Recording Duration (seconds)", min_value=2, max_value=10, value=5)
ground_truth = st.text_input("Enter Ground Truth (what you actually said) for accuracy calculation:")

if st.button("🎤 Record and Recognize"):
    if ground_truth.strip() == "":
        st.warning("Please enter the ground truth text to compute accuracy/WER.")
    else:
        audio_file = record_audio(duration=duration)
        st.audio(audio_file, format="audio/wav")
        st.info("Processing audio...")
        comparison_df = compare_methods(audio_file, ground_truth)
        
        st.subheader("🔎 Comparison Table with WER and Accuracy")
        st.table(comparison_df)
        
        st.subheader("📌 Observations")
        st.write("- Google API is generally accurate with fast/noisy speech but needs internet.")
        st.write("- Vosk works offline but may struggle with unclear or soft speech.")
        st.write("- Whisper works offline and is usually the most accurate offline method.")
        st.write("- WER closer to 0 and Accuracy closer to 100% = better recognition.")
