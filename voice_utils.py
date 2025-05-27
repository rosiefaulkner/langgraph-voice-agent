from dotenv import load_dotenv
import io
import logging
import numpy as np
import sounddevice as sd
from scipy.io.wavfile import write
from langchain_google_genai import ChatGoogleGenerativeAI
import asyncio
import os

load_dotenv()

# Configure logging
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("langchain_google_genai").setLevel(logging.WARNING)

# Initialize Gemini speech API
gemini = ChatGoogleGenerativeAI(
    model="gemini-pro",
    google_api_key=os.getenv("GOOGLE_API_KEY")
)

async def record_audio_until_stop():
    """Records audio from the microphone until Enter is pressed, then saves it to a .wav file."""

    audio_data = []
    recording = True
    sample_rate = 16000

    def record_audio():
        nonlocal audio_data, recording
        with sd.InputStream(samplerate=sample_rate, channels=1, dtype='int16') as stream:
            while recording:
                audio_chunk, _ = stream.read(1024)
                audio_data.append(audio_chunk)

    def stop_recording():
        input()  # Press Enter to stop
        nonlocal recording
        recording = False

    loop = asyncio.get_running_loop()
    stop_task = loop.run_in_executor(None, stop_recording)
    record_task = loop.run_in_executor(None, record_audio)

    await stop_task
    await record_task

    audio_data = np.concatenate(audio_data, axis=0)
    audio_bytes = io.BytesIO()
    write(audio_bytes, sample_rate, audio_data)
    audio_bytes.seek(0)
    audio_bytes.name = "audio.wav"

    # TODO: Replace with Gemini-compatible audio transcription 
    # As a placeholder, return a mock transcription:
    print("⚠️ Gemini does not support audio transcription. Using dummy text.")
    transcription_text = "This is a mock transcription."

    return transcription_text


async def play_audio(message: str):
    """Plays the assistant's response using an external TTS system."""
    cleaned_message = message.replace("**", "")
    
    # TODO: Replace with Google Cloud Text-to-Speech API or another TTS engine here.
    print("\n🗣️ (Simulated Speech Playback):", cleaned_message)
    await asyncio.sleep(0.5) 
