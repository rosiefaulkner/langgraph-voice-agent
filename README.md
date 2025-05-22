# Luna: Voice-Enabled Expense Management Agent

Luna is a voice-enabled AI assistant built with Langgraph that helps users manage their expenses through natural conversation. This project demonstrates how to create a voice interface for any Langgraph agent, combining speech-to-text and text-to-speech capabilities with a powerful agent framework.

## 🌟 Features

- **Voice Interaction**: Speak to Luna and hear responses through high-quality text-to-speech
- **Expense Management**: Create, query, update, and delete expenses through natural conversation
- **Category Classification**: Automatically categorizes expenses based on descriptions
- **Database Integration**: Stores expense data in a PostgreSQL database (via Supabase)
- **Tool-using Agent**: Built with Langgraph's agent framework for complex reasoning

## 🛠️ Technology Stack

### Backend

- **Python 3.13**: Core language for the backend
- **Langgraph**: Agent framework for building the conversational AI
- **OpenAI**:
  - Whisper API for speech-to-text
  - GPT-4 Mini for the agent's reasoning
  - TTS API for text-to-speech responses
- **MCP (Model Calling Protocol)**: For defining and using tools
- **SQLAlchemy**: ORM for database interactions
- **Supabase**: PostgreSQL database provider

### Audio Processing

- **sounddevice**: For capturing audio from microphone
- **scipy**: For audio file processing

## 📋 Prerequisites

- Python 3.13
- OpenAI API key
- Supabase account and database
- Microphone and speakers

## 🚀 Getting Started

### 1. Clone the repository

```bash
git clone https://github.com/yourusername/langgraph-voice-agent.git
cd langgraph-voice-agent
```

### 2. Set up a virtual environment and install dependencies

(Recommended) use [uv](https://github.com/uvlabs/uv) for dependency management

Setup the venv in your project directory and install all dependencies with one command.

```bash
uv sync
```

### 3. Set up environment variables

Create a `.env` file in the root directory with the following variables:

```env
OPENAI_API_KEY=your_openai_api_key
SUPABASE_URI=postgresql://postgres:password@db.example.supabase.co:5432/postgres
```

### 4. Run the application

```bash
python main.py
```

## 🎤 Using Luna

1. Run the application
2. When prompted, speak your request (e.g., "Create a new expense for lunch today that cost $15")
3. Press Enter to stop recording
4. Luna will process your request, interact with the database if needed, and respond verbally
5. Continue the conversation or say "exit" or "quit" to end the session

## 🧩 Project Structure

```plaintext
langgraph-voice-agent/
├── main.py                  # Main application entry point
├── assistant_graph.py       # Langgraph agent definition
├── state.py                 # State management for the agent
├── voice_utils.py           # Audio recording and playback utilities
├── mcps/                    # Model Calling Protocol servers
│   ├── mcp_config.json      # MCP server configuration
│   └── local_servers/
│       └── db.py            # Database tools implementation
├── .env                     # Environment variables (not in repo)
├── .env.example             # Example environment variables
└── pyproject.toml           # Project dependencies
```

## 🔧 Customizing the Agent

### Modifying the System Prompt

To change Luna's personality or capabilities, edit the `system_prompt` in `assistant_graph.py`:

```python
system_prompt = """You are Luna, the company's expense manager...
```

### Adding New Tools

1. Create a new MCP server or add tools to the existing one in `mcps/local_servers/`
2. Register the server in `mcps/mcp_config.json`
3. The tools will be automatically available to the agent

### Changing Voice Settings

Modify the TTS settings in `voice_utils.py`:

```python
async def play_audio(message: str):
    # ...
    async with openai_async.audio.speech.with_streaming_response.create(
        model="gpt-4o-mini-tts",
        voice="fable",  # Change the voice here
        input=cleaned_message,
        instructions="Speak in a cheerful, helpful tone with a brisk pace.",  # Modify instructions
        response_format="pcm",
        speed=1.2,  # Adjust speed
    ) as response:
        # ...
```

## 📚 Learning Resources

- [Intro to Langgraph](https://youtu.be/31JoTDm7jkM)
- [Deploy Langgraph Agents](https://youtu.be/SGt786ne_Mk)
- [MCP with Langgraph Agents](https://youtu.be/F9mgEFor0cA)
- [Langgraph Documentation](https://python.langchain.com/docs/langgraph/)
- [OpenAI API Documentation](https://platform.openai.com/docs/introduction)
- [Model Context Protocol (MCP)](https://modelcontextprotocol.io/introduction)
- [SQLAlchemy Documentation](https://docs.sqlalchemy.org/)

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.
