from dotenv import load_dotenv
import io
import logging
import numpy as np
import sounddevice as sd
from scipy.io.wavfile import write
from openai import OpenAI, AsyncOpenAI
from openai.helpers import LocalAudioPlayer
import asyncio


load_dotenv()

# Configure logging to suppress HTTP request logs
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("openai").setLevel(logging.WARNING)

openai_async = AsyncOpenAI()
openai = OpenAI()


async def record_audio_until_stop():
    """Records audio from the microphone until Enter is pressed, then saves it to a .wav file."""

    audio_data = []  # List to store audio chunks
    recording = True  # Flag to control recording
    sample_rate = 16000 # (kHz) Adequate for human voice frequency

    def record_audio():
        """Continuously records audio until the recording flag is set to False."""
        nonlocal audio_data, recording
        # This sounddevice stream read is blocking, so it will run in the executor thread
        with sd.InputStream(samplerate=sample_rate, channels=1, dtype='int16') as stream:
            while recording:
                audio_chunk, _ = stream.read(1024)  # Read audio data in chunks
                audio_data.append(audio_chunk)

    def stop_recording():
        """Waits for user input to stop the recording."""
        # This input() call is blocking, so it will run in the executor thread
        input()  # Wait for Enter key press
        nonlocal recording
        recording = False

    loop = asyncio.get_running_loop()

    # Run the blocking functions in the event loop's default executor
    stop_task = loop.run_in_executor(None, stop_recording)
    record_task = loop.run_in_executor(None, record_audio)

    # Wait for both tasks to complete
    await stop_task
    await record_task

    # Stack all audio chunks into a single NumPy array and write to file
    audio_data = np.concatenate(audio_data, axis=0)

    # Convert to WAV format in-memory
    audio_bytes = io.BytesIO()
    # Use scipy's write function to save to BytesIO (this is fast, doesn't need executor)
    write(audio_bytes, sample_rate, audio_data)
    audio_bytes.seek(0)  # Go to the start of the BytesIO buffer
    audio_bytes.name = "audio.wav" # Set a filename for the in-memory file

    # Transcribe via Whisper (async call, no need for executor)
    transcription = await openai_async.audio.transcriptions.create(
        model="whisper-1",
        file=audio_bytes,
    )

    return transcription.text


async def play_audio(message: str):
    """Plays the audio response from the remote graph with OpenAI."""
    cleaned_message = message.replace("**", "")

    async with openai_async.audio.speech.with_streaming_response.create(
        model="gpt-4o-mini-tts",
        voice="fable",
        input=cleaned_message,
        instructions="Speek in a cheerful, helpful tone with a brisk pace.",
        response_format="pcm",
        speed=1.2,
    ) as response:
        await LocalAudioPlayer().play(response)
