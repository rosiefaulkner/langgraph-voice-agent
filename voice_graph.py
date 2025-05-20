from langchain_core.messages import HumanMessage
from langgraph.graph import StateGraph
from langgraph.graph.state import CompiledStateGraph
from langgraph.checkpoint.memory import InMemorySaver
from dotenv import load_dotenv
import io
import threading
import numpy as np
import sounddevice as sd
from scipy.io.wavfile import write
from langgraph.prebuilt import ToolNode
from openai import OpenAI, AsyncOpenAI
from openai.helpers import LocalAudioPlayer
from state import AgentState
from assistant_graph import Agent


load_dotenv()

openai_async = AsyncOpenAI()
openai = OpenAI()


def build_voice_graph() -> CompiledStateGraph:
    def record_audio_until_stop(state: AgentState):
        """Records audio from the microphone until Enter is pressed, then saves it to a .wav file."""
        
        audio_data = []  # List to store audio chunks
        recording = True  # Flag to control recording
        sample_rate = 16000 # (kHz) Adequate for human voice frequency
        stop_event = threading.Event() # Event to signal threads to stop

        def record_audio(stop_event):
            """Continuously records audio until the recording flag is set to False."""
            nonlocal audio_data, recording
            with sd.InputStream(samplerate=sample_rate, channels=1, dtype='int16') as stream:
                print("Recording your instruction! ... Press Enter to stop recording.", flush=True)
                while not stop_event.is_set():
                    audio_chunk, _ = stream.read(1024)  # Read audio data in chunks
                    audio_data.append(audio_chunk)
                    if stop_event.is_set():
                        break

        def stop_recording(stop_event):
            """Waits for user input to stop the recording."""
            user_input = input()  # Wait for Enter key press
            if user_input == "":
                nonlocal recording
                recording = False 
                stop_event.set() 

        # Start recording in a separate thread
        recording_thread = threading.Thread(target=record_audio, args=(stop_event,))
        recording_thread.start()

        # Start a thread to listen for the Enter key
        stop_thread = threading.Thread(target=stop_recording, args=(stop_event,))
        stop_thread.start()

        # Wait for both threads to complete
        stop_thread.join()
        recording_thread.join()

        # Stack all audio chunks into a single NumPy array and write to file
        audio_data = np.concatenate(audio_data, axis=0)
        
        # Convert to WAV format in-memory
        audio_bytes = io.BytesIO()
        write(audio_bytes, sample_rate, audio_data)  # Use scipy's write function to save to BytesIO
        audio_bytes.seek(0)  # Go to the start of the BytesIO buffer
        audio_bytes.name = "audio.wav" # Set a filename for the in-memory file

        # Transcribe via Whisper
        transcription = openai.audio.transcriptions.create(
        model="whisper-1", 
        file=audio_bytes,
        )

        # Print the transcription
        print("Here is the transcription:", transcription.text)

        # Write to messages 
        state.messages.append(HumanMessage(content=transcription.text))
        return state

    async def play_audio(state: AgentState):
        
        """Plays the audio response from the remote graph with ElevenLabs."""

        # Response from the agent 
        last_message = state.messages[-1]
        cleaned_message = last_message.content.replace("**", "")
        
        async with openai_async.audio.speech.with_streaming_response.create(
            model="gpt-4o-mini-tts",
            voice="fable",
            input=cleaned_message,
            instructions="Speek in a cheerful, helpful tone.",
            response_format="pcm",
        ) as response:
            await LocalAudioPlayer().play(response)

    builder = StateGraph(AgentState)
    builder.add_node("audio_input", record_audio_until_stop)
    builder.add_node("assistant", Agent().build_graph())
    builder.add_node("audio_output", play_audio)

    builder.set_entry_point("audio_input")
    builder.add_edge("audio_input", "assistant")
    builder.add_edge("assistant", "audio_output")
    builder.set_finish_point("audio_output")

    return builder.compile(checkpointer=InMemorySaver())

voice_graph = build_voice_graph()

if __name__ == "__main__":
    from IPython.display import Image
    
    Image(voice_graph.get_graph(xray=True).draw_mermaid_png())

    config = {"configurable": {"thread_id": "thread-1"}}
    voice_graph.invoke(AgentState(messages=[
        HumanMessage(content="Hello")
    ]), config=config)
