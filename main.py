from langchain_core.messages import AIMessage, AIMessageChunk, HumanMessage
from langgraph.graph import StateGraph
from typing import AsyncGenerator
import asyncio

from state import AgentState
from voice_graph import voice_graph
from assistant_graph import agent_graph
from tools import session


async def stream_graph_response(
        input: AgentState, graph: StateGraph, config: dict = {}
        ) -> AsyncGenerator[str, None]:
    """
    Stream the response from the graph while parsing out tool calls.

    Args:
        input: The input for the graph.
        graph: The graph to run.
        config: The config to pass to the graph. Required for memory.

    Yields:
        A processed string from the graph's chunked response.
    """
    async for message_chunk, metadata in graph.astream(
        input=input,
        stream_mode="messages",
        config=config
        ):
        if isinstance(message_chunk, AIMessageChunk):
            if message_chunk.response_metadata:
                finish_reason = message_chunk.response_metadata.get("finish_reason", "")
                if finish_reason == "tool_calls":
                    yield "\n\n"

            if message_chunk.tool_call_chunks:
                tool_chunk = message_chunk.tool_call_chunks[0]

                tool_name = tool_chunk.get("name", "")
                args = tool_chunk.get("args", "")
                
                if tool_name:
                    tool_call_str = f"\n\n< TOOL CALL: {tool_name} >\n\n"
                if args:
                    tool_call_str = args

                yield tool_call_str
            else:
                yield message_chunk.content
            continue

async def main():
    config = {"configurable": {"thread_id": "thread-1"}}
    customer_id = "e6535c6e-d9a4-4e95-a100-8224494fa01f"

    # Get tools from our MCP server
    client = MultiServerMCPClient(connections=mcp_config["mcpServers"])
    tools = await client.get_tools()

    agent_graph = Agent(tools=tools).build_graph()

    # Initialize the input state outside the loop for the first turn
    initial_input = AgentState(customer_id=customer_id)

    while True:
        # Record audio, transcribe, and add the human message to the state
        print("\n\nSpeak now, then press Enter to stop recording...")
        transcribed_text = await record_audio_until_stop()
        initial_input.messages.append(HumanMessage(content=transcribed_text))

        # check exit condition
        if transcribed_text.lower().count("exit") or transcribed_text.lower().count("quit"):
            print("\nExit command received. Ending conversation.")
            break

        print("\n ---- You ---- \n\n", transcribed_text, "\n")

        print("\n ---- Assistant ---- ")
        async for response in stream_graph_response(
            input = initial_input,
            graph = agent_graph,
            config = config
            ):
            print(response, end="", flush=True)

        # Get the latest state
        thread_state = agent_graph.get_state(config=config)
        
        # Play the assistant's response
        last_message = thread_state.values.get("messages")[-1]
        if isinstance(last_message, AIMessage):
            await play_audio(last_message.content)


if __name__ == "__main__":
    import asyncio
    import nest_asyncio
    nest_asyncio.apply()
    asyncio.run(main())
